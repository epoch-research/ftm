import scipy.stats
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import statsmodels.distributions.copula.api as sm_api
import statsmodels.distributions.copula.copulas as sm_copulas

from ..core.utils import log, get_parameter_table, get_rank_correlations, get_clipped_ajeya_dist

class TakeoffParamsDist():
  """ Joint parameter distribution. """

  def __init__(self, ensure_no_automatable_goods_tasks = True, ignore_rank_correlations = False, use_ajeya_dist = True, resampling_method = 'gap_only',
          parameter_table = None):
    """
      If ensure_no_automatable_goods_tasks is True, we'll make sure none of the samples
      represent an scenario in which there is some "goods" task initially automatable.
    """

    self.resampling_method = resampling_method

    # Retrieve parameter table
    if parameter_table is None:
      log.info('Retrieving parameters...')
      parameter_table = get_parameter_table()
      parameter_table = parameter_table[['Conservative', 'Best guess', 'Aggressive', 'Type']]

      # By defaul, disable the runtime-training tradeoff
      parameter_table.at['runtime_training_tradeoff', 'Conservative'] = None
      parameter_table.at['runtime_training_tradeoff', 'Best guess']   = 0
      parameter_table.at['runtime_training_tradeoff', 'Aggressive']   = None

      parameter_table.at['runtime_training_max_tradeoff', 'Conservative'] = None
      parameter_table.at['runtime_training_max_tradeoff', 'Best guess']   = 1
      parameter_table.at['runtime_training_max_tradeoff', 'Aggressive']   = None

    self.parameter_table = parameter_table

    self.rank_correlations = get_rank_correlations()

    self.marginals = self.get_marginals(parameter_table, use_ajeya_dist)

    if ignore_rank_correlations:
      self.pairwise_rank_corr = {}
    else:
      marginal_directions = self.get_marginal_directions(parameter_table)
      self.pairwise_rank_corr = self.process_correlations(self.marginals, self.rank_correlations, marginal_directions)

    self.joint_dist = JointDistribution(self.marginals, self.pairwise_rank_corr, rank_corr_method = "spearman")
    self.ensure_no_automatable_goods_tasks = ensure_no_automatable_goods_tasks

  def get_marginal_directions(self, parameter_table):
    directions = {}
    for parameter, row in parameter_table.iterrows():
      if not np.isnan(row['Conservative']) and not np.isnan(row['Aggressive']):
        directions[parameter] = +1 if (row['Conservative'] < row['Aggressive']) else -1
    return directions

  def get_marginals(self, parameter_table, use_ajeya_dist):
    marginals = {}
    for parameter, row in parameter_table.iterrows():
      if not np.isnan(row['Conservative']) and not np.isnan(row['Aggressive']):
        marginal = SkewedLogUniform(
          row['Conservative'],
          row['Best guess'],
          row['Aggressive'],
          kind = row['Type']
        )
      else:
        marginal = PointDistribution(row['Best guess'])
      marginals[parameter] = marginal

    lowest_training_requirements_goods = \
        (marginals['initial_biggest_training_run'].b * marginals['runtime_training_max_tradeoff'].b) \
        * marginals['flop_gap_training'].a**(10.5/7)

    lowest_training_requirements_rnd = \
        (marginals['initial_biggest_training_run'].b * marginals['runtime_training_max_tradeoff'].b * marginals['goods_vs_rnd_requirements_training'].b) \
        * marginals['flop_gap_training'].a**(10.5/7)

    lowest_training_requirements = max(lowest_training_requirements_goods, lowest_training_requirements_rnd)

    if use_ajeya_dist:
      marginals['full_automation_requirements_training'] = \
          AjeyaDistribution(lower_bound = lowest_training_requirements)
    else:
      marginals['full_automation_requirements_training'] = SkewedLogUniform(\
        parameter_table.at['full_automation_requirements_training', 'Conservative'],
        parameter_table.at['full_automation_requirements_training', 'Best guess'],
        max(lowest_training_requirements, parameter_table.at['full_automation_requirements_training', 'Aggressive']),
      )

    return marginals

  def process_correlations(self, marginals, rank_correlations_matrix, directions):
    correlations = {}
    for left in marginals.keys():
      for right in marginals.keys():
        if right not in rank_correlations_matrix or left not in rank_correlations_matrix:
          continue

        if isinstance(marginals[right], PointDistribution) or isinstance(marginals[left], PointDistribution):
          continue

        r = rank_correlations_matrix[right][left]
        if not np.isnan(r) and r != 0:
          correlations[(left, right)] = r * directions[left]*directions[right]
    return correlations

  def rvs(self, count, random_state = None, conditions = {}, resampling_method = None):
    # statsmodels.distributions.copula.copulas throws an exception when we ask less than 2 samples from it.
    # We could make this more efficient, but it's probably not worth it.

    if resampling_method is None: resampling_method = self.resampling_method

    if resampling_method == 'gap_only':
      # statsmodels.distributions.copula.copulas throws an exception when we ask less than 2 samples from it
      actual_count = max(count, 2)
      samples = self.joint_dist.rvs(actual_count, random_state = random_state, conditions = conditions)[:count]

      if self.ensure_no_automatable_goods_tasks:
        # Resample the training gap to ensure no tasks is automatable from the beginning
        gap_marginal = self.marginals['flop_gap_training']
        for i, row in samples.iterrows():
          max_gap = (row['full_automation_requirements_training']/(row['initial_biggest_training_run'] * row['runtime_training_max_tradeoff']))**(7/10.5)
          gap_marginal.set_upper_bound(max_gap)
          samples.at[i, 'flop_gap_training'] = gap_marginal.rvs(random_state = random_state)
        gap_marginal.set_upper_bound(None)

      return samples
    else:
      output_samples = pd.DataFrame(columns = [name for name in self.marginals], index = range(count), dtype = np.float64)
      output_samples_count = 0

      while output_samples_count < count:
        samples_to_read = count - output_samples_count

        # statsmodels.distributions.copula.copulas throws an exception when we ask less than 2 samples from it
        # (hence the max)
        samples = self.joint_dist.rvs(max(2, samples_to_read), random_state = random_state, conditions = conditions)[:samples_to_read]

        for i, sample in samples.iterrows():
          # Ensure the sample makes sense before adding it
          if self.ensure_no_automatable_goods_tasks:
            max_gap_goods = \
              (sample['full_automation_requirements_training']/(sample['initial_biggest_training_run'] * sample['runtime_training_max_tradeoff']))**(7/10.5)
            max_gap_rnd = \
              max_gap_goods/sample['goods_vs_rnd_requirements_training']**(7/10.5)
            max_gap = min(max_gap_rnd, max_gap_goods)

            if sample['flop_gap_training'] >= max_gap:
              continue

          # OK, looks good
          output_samples.loc[output_samples_count] = samples.loc[i]
          output_samples_count += 1

      return output_samples

class AjeyaDistribution(rv_continuous):
  def __init__(self, lower_bound = None):
    self.cdf_pd = get_clipped_ajeya_dist(lower_bound)

    cdf = self.cdf_pd.to_numpy()
    self.v = cdf[:, 0]
    self.p = cdf[:, 1]

    super().__init__(a = 10**np.min(self.v), b = 10**np.max(self.v))

  def _cdf(self, v):
    p = interp1d(self.v, self.p)(np.log10(v))
    return p

  def _ppf(self, p):
    # SciPy has a hard time computing the PPF from the CDF, so we are doing it ourselves
    return 10**interp1d(self.p, self.v)(p)

class GaussianCopula(sm_api.GaussianCopula):
  def __init__(self, corr=None, k_dim=2):
    super().__init__(corr=corr, k_dim=k_dim)
    self.mu = np.zeros(len(corr))

  def rvs(self, nobs=1, args=[], random_state=None):
    # The "0.5 + (1 - 1e-10) * (x - 0.5)" below is to ensure we pass to the normal ppf only values inside (0, 1).
    # TODO: Is this reasonable? sm_copulas.CopulaDistribution does the same
    fixed_values = [self.distr_uv.ppf(0.5 + (1 - 1e-10) * (x - 0.5)) for x in args]
    x = self.sample_normal_cond(values = fixed_values, nobs = nobs, random_state = random_state)
    return self.distr_uv.cdf(x)

  def sample_normal_cond(self, values=[], nobs=1, random_state=None):
    # NOTE: Code by Ege (with some minor modifications)

    S = []
    D = []
    a = []

    values = np.array(values)
    for count, q in enumerate(values):
      if not np.isnan(q):
        S.append(count)
        a.append(q)
      else:
        D.append(count)
    
    if S == []:
      return multivariate_normal.rvs(cov=self.corr, size=nobs, random_state=random_state)
    elif D == []:
      return []
    else:
      D = np.array(D, dtype=int)
      S = np.array(S, dtype=int)
      a = np.array(a)

      mu_1 = self.mu[D]
      mu_2 = self.mu[S]

      cov_11 = self.corr[D, :][:, D]
      cov_12 = self.corr[D, :][:, S]
      cov_21 = self.corr[S, :][:, D]
      cov_22 = self.corr[S, :][:, S]

      mu_bar = mu_1 + np.dot(np.matmul(cov_12, np.linalg.inv(cov_22)), a - mu_2)
      cov_bar = cov_11 - np.matmul(np.matmul(cov_12, np.linalg.inv(cov_22)), cov_21)

      samples = []
      for s in multivariate_normal.rvs(mean=mu_bar, cov=cov_bar, size=nobs, random_state=random_state):
        sample = values.copy()
        sample[D] = s
        samples.append(sample)

      return samples

# NOTE: Code taken from https://github.com/tadamcz/copula-wrapper/blob/main/copula_wrapper/correlation_convert.py
def get_pearsons_rho(kendall=None, spearman=None):
    """
    Bivariate Pearson's rho from bivariate rank correlations (Kendall's tau or Spearman's rho).

    References: https://www.mathworks.com/help/stats/copulas-generate-correlated-samples.html
    """

    if kendall is not None and spearman is not None:
        raise ValueError("Must provide exactly one of `kendall` or `spearman`.")

    if kendall is not None:
        func = lambda kendalls_tau: np.sin(kendalls_tau * np.pi / 2)
        arg = kendall
    elif spearman is not None:
        func = lambda spearmans_rho: 2 * np.sin(spearmans_rho * np.pi / 6)
        arg = spearman
    else:
        raise ValueError("Must provide exactly one of `kendall` or `spearman`.")

    try:
        return func(arg)
    except TypeError:
        return np.vectorize(func)(arg)

# NOTE: Code taken from https://github.com/tadamcz/copula-wrapper/blob/main/copula_wrapper/joint_distribution.py with some additions
class JointDistribution(scipy.stats.rv_continuous):
    """
    Thin wrapper around `CopulaDistribution` from `statsmodels`.

    Currently, this exclusively uses the Gaussian copula.

    The main differences are:
    - This interface explicitly requires a rank correlation, instead of the Pearson's rho of the transformed variables.
    - Dimensions require names (given as dictionary keys) and must be accessed by their names (as keyword arguments to
    .pdf, .cdf, and .logpdf), instead of their indices.
    - .rvs returns samples as a Pandas DataFrame with dimension names as column names.

    todo:
        - improve subclassing of scipy.rv_continuous. Current approach is quick and dirty, and fails for some
        methods that should work, like .ppf. This is because you're supposed to override the underscored methods ._pdf,
        ._cdf, etc., instead of the methods .pdf, .cdf, etc. I haven't yet figured out how override these well for an
        n-dimensional distribution.
    """

    def __init__(self, marginals, rank_corr, rank_corr_method):
        """
        :param marginals: Dictionary of size `n`, where the keys are dimension names as strings and the values are
        SciPy continuous distributions.

        :param rank_corr: Dictionary of pairwise rank correlations. Missing pairs are
        assumed to be independent.

        :param rank_corr_method: 'spearman' for Spearman's rho, or 'kendall' for Kendall's tau.
        """
        super().__init__()
        self.marginals = marginals
        self.rank_correlation = rank_corr
        self.rank_correlation_method = rank_corr_method

        self.dimension_names = {}
        marginals_list = [None] * len(marginals)
        for index, (name, distribution) in enumerate(marginals.items()):
            self.dimension_names[name] = index
            marginals_list[index] = distribution

        rank_corr_matrix = self._to_matrix(rank_corr)

        if rank_corr_method == 'spearman':
            pearsons_rho = get_pearsons_rho(spearman=rank_corr_matrix)
        elif rank_corr_method == 'kendall':
            pearsons_rho = get_pearsons_rho(kendall=rank_corr_matrix)
        else:
            raise ValueError("`rank_corr_method` must be one of 'spearman' or 'kendall'")

        # `pearsons_rho` refers to the correlations of the Gaussian-transformed variables
        copula_instance = GaussianCopula(corr=pearsons_rho)
        self.wrapped = sm_copulas.CopulaDistribution(copula_instance, marginals_list)
        self.wrapped.rank_correlation = rank_corr_matrix

    def rvs(self, nobs=2, random_state=None, conditions={}):
        # Get conditional values
        fixed_values = [marginal.cdf(conditions.get(name, np.nan)) for (name, marginal) in self.marginals.items()]

        as_df = pd.DataFrame()

        if np.any(np.isnan(fixed_values)):
          rvs = self.wrapped.rvs(nobs=nobs, random_state=random_state, cop_args=fixed_values)
          for name, i in self.dimension_names.items():
              column = rvs[:, i]
              as_df[name] = column
        else:
          for name, i in self.dimension_names.items():
              column = np.full(nobs, conditions[name])
              as_df[name] = column

        return as_df

    def cdf(self, **kwargs):
        return self.wrapped.cdf(self._to_tuple(kwargs))

    def pdf(self, **kwargs):
        return self.wrapped.pdf(self._to_tuple(kwargs))

    def logpdf(self, **kwargs):
        return self.wrapped.logpdf(self._to_tuple(kwargs))

    def sf(self, **kwargs):
        return 1 - self.cdf(**kwargs)

    def _to_tuple(self, kwargs):
        if kwargs.keys() != self.dimension_names.keys():
            raise ValueError(f"You must provide the following keyword arguments: {list(self.dimension_names.keys())}")
        iterable = [None] * len(self.marginals)
        for name, index in self.dimension_names.items():
            iterable[index] = kwargs[name]
        return tuple(iterable)

    def _to_matrix(self, rank_correlation):
        corr_matrix = np.eye(N=len(self.marginals))
        names_to_indices = self.dimension_names
        for index, (pair, correlation) in enumerate(rank_correlation.items()):
            left, right = pair
            i, j = names_to_indices[left], names_to_indices[right]
            corr_matrix[i][j] = correlation
            corr_matrix[j][i] = correlation
        return corr_matrix

class SkewedLogUniform(rv_continuous):
  def __init__(self, low, med, high, kind = 'pos'):
    if high < low:
      low, high = high, low

    super().__init__(a = low, b = high)

    self.upper_bound = high
    self.initial_upper_bound = high

    # Transform to loguniform
    if kind == "frac":
      low = low / (1. - low)
      med = med / (1. - med)
      high = high / (1. - high)

    elif kind == 'neg':
      low = -low
      med = -med
      high = -high

    elif kind == "inv_frac":
      low = 1/low
      med = 1/med
      high = 1/high

      low = low / (1. - low)
      med = med / (1. - med)
      high = high / (1. - high)

    elif not kind == "pos":
      raise ValueError(f"Unimplemented kind: {kind}")

    # Apply log to transform to uniform
    low = np.log(low)
    med = np.log(med)
    high = np.log(high)

    self.low = low
    self.med = med
    self.high = high
    self.kind = kind

    self.integration_direction = +1 if (low < high) else -1

    self._cdf = np.vectorize(self._cdf)

  def set_upper_bound(self, bound):
    if bound is None: bound = self.initial_upper_bound
    bound = min(bound, self.initial_upper_bound)
    bound = max(bound, self.a)
    self.upper_bound = bound
    self.b = bound

  def _cdf(self, x):
    # Transform to loguniform

    if self.kind == "frac":
      y = x / (1. - x)
    elif self.kind == "inv_frac":
      y = 1 / x
      y = y / (1. - y)
    elif self.kind == 'neg':
      y = -x
    else:
      y = x

    # Apply log to transform to uniform
    y = np.log(y)

    s = self.integration_direction

    if s*y < s*self.low:
      q = 0
    elif s*y < s*self.med:
      q = 1./2 * (y - self.low) / (self.med - self.low)
    elif s*y < s*self.high:
      q = 1./2 + 1./2 * (y - self.med) / (self.high - self.med)
    else:
      q = 1

    return q

  def _ppf(self, q):
    upper_q = self._cdf(self.upper_bound)
    q = q * upper_q

    scalar_input = np.isscalar(q)

    if scalar_input:
      q = np.array([q])

    y = np.zeros(len(q))

    for i in range(len(q)):
      if q[i] <= 0:
        y[i] = self.low
      elif q[i] <= 1./2:
        y[i] = self.low + 2 * q[i] * (self.med - self.low)
      elif q[i] < 1:
        y[i] = self.med + 2 * (q[i] - 1./2) * (self.high - self.med)
      else:
        y[i] = self.high

    y = np.exp(y)

    if self.kind == "frac":
      x = y / (1. + y)
    elif self.kind == "inv_frac":
      x = y / (1. + y)
      x = 1 / x
    elif self.kind == 'neg':
      x = -y
    else:
      x = y

    return x[0] if scalar_input else x

class PointDistribution(rv_continuous):
  def __init__(self, v):
    self.v = v
    super().__init__(a = v, b = v)

  def _ppf(self, q):
    return self.v

  def get_value(self):
    return self.v
