"""
Monte carlo analysis.
"""

from . import log
from . import *

import pickle
import traceback
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous
from scipy.special import erfinv
from numpy.random import default_rng
from matplotlib import cm
from xml.etree import ElementTree as et
from copula_wrapper import joint_distribution
from ..core.utils import get_clipped_ajeya_dist, get_param_names, get_metric_names

rng = default_rng()

class McAnalysisResults:
  pass

class TooManyRetries(Exception):
  pass

def mc_analysis(n_trials = 100, max_retries = 100):
  scalar_metrics = ['rampup_start', 'agi_year']
  state_metrics = ['biggest_training_run']

  scalar_metrics = {
      scalar_metric : [] for scalar_metric in scalar_metrics
  }
  for takeoff_metric in SimulateTakeOff.takeoff_metrics:
    scalar_metrics[takeoff_metric] = []

  state_metrics = {
      state_metric : [] for state_metric in state_metrics
  }

  slow_takeoff_count = 0

  params_dist = ParamsDistribution()
  samples = []

  last_valid_indices = []

  log.info(f'Running simulations...')
  log.indent()
  for trial in range(n_trials):
    for i in range(max_retries):
      # Try to run the simulation
      try:
        log.info(f'Running simulation {trial+1}/{n_trials}...')

        sample = params_dist.rvs(1)
        mc_params = {param: sample[param][0] for param in sample}

        mc_model = SimulateTakeOff(**mc_params)

        # Check that no goods task is automatable from the beginning (except for the first one)
        runtime_training_max_tradeoff = mc_model.runtime_training_max_tradeoff if mc_model.runtime_training_tradeoff is not None else 1.
        assert(not np.any(mc_model.automation_training_flops_goods[1:] < mc_model.initial_biggest_training_run * runtime_training_max_tradeoff))

        mc_model.run_simulation()
      except Exception as e:
        # This was a bad sample. We'll just discard it and try again.
        log.indent()
        log.info('The model threw an exception:')
        log.indent()
        log.info(e)
        log.info(traceback.format_exc(), end = '')
        log.deindent()
        log.info('Discarding the sample and rerunning the simulation')
        log.deindent()
        continue

      # This was a good sample
      samples.append(sample)
      break
    else:
      raise TooManyRetries('MC sampling: Maximum number of retries reached')

    # Collect results
    for scalar_metric in scalar_metrics:
      if scalar_metric in SimulateTakeOff.takeoff_metrics:
        metric_value = mc_model.takeoff_metrics[scalar_metric]
        assert metric_value >= 0, f"{scalar_metric} is negative!"
      else:
        metric_value = getattr(mc_model, scalar_metric)

      if scalar_metric == "rampup_start" and metric_value is None:
        metric_value = mc_model.t_end
      elif scalar_metric == "agi_year" and metric_value is None:
        metric_value = mc_model.t_end
      scalar_metrics[scalar_metric].append(metric_value)

    for state_metric in state_metrics:
      metric_value = getattr(mc_model, state_metric)
      assert metric_value.shape == (mc_model.n_timesteps,)
      state_metrics[state_metric].append(metric_value)

    last_valid_indices.append(mc_model.t_idx)

    if is_slow_takeoff(mc_model):
      slow_takeoff_count += 1

  log.deindent()

  # Summary of scalar metrics
  quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
  metrics_quantiles = []
  for q in quantiles:
    row = {"quantile" : q}
    for scalar_metric in scalar_metrics:
      row[scalar_metric] = np.quantile(scalar_metrics[scalar_metric], q)
    metrics_quantiles.append(row)

  ## Add mean
  row = {"quantile" : "mean"}
  for scalar_metric in scalar_metrics:
    row[scalar_metric] = np.mean(scalar_metrics[scalar_metric])
  metrics_quantiles.append(row)

  results = McAnalysisResults()
  results.quantiles          = quantiles
  results.metrics_quantiles  = metrics_quantiles
  results.state_metrics      = state_metrics
  results.scalar_metrics     = scalar_metrics
  results.n_trials           = n_trials
  results.timesteps          = mc_model.timesteps
  results.t_step             = mc_model.t_step
  results.param_samples      = pd.concat(samples, ignore_index = True)
  results.ajeya_cdf          = params_dist.marginals['full_automation_requirements_training'].cdf_pd
  results.parameter_table    = params_dist.parameter_table
  results.rank_correlations  = params_dist.rank_correlations
  results.slow_takeoff_count = slow_takeoff_count
  results.last_valid_indices = last_valid_indices

  return results

def is_slow_takeoff(model):
  return n_year_doubling_before_m_year_doubling(model.gwp[:model.t_idx], model.t_step, 4, 1)

def n_year_doubling_before_m_year_doubling(array, t_step, n, m):
  delta_n = round(n/t_step)
  delta_m = round(m/t_step)

  idx_n = SimulateTakeOff.first_index(array[delta_n:]/array[:-delta_n] >= 2)
  idx_m = SimulateTakeOff.first_index(array[delta_m:]/array[:-delta_m] >= 2)

  if idx_m is None:
    return True
  elif idx_n is None:
    return False

  t_diff = (idx_m - idx_n) * t_step

  return t_diff >= n

def write_takeoff_probability_table(n_trials=100, max_retries=100, input_results_filename=None):
  if input_results_filename:
    with open(input_results_filename, 'rb') as f:
      results = pickle.load(f)
  else:
    results = mc_analysis(n_trials, max_retries)

  t_step = results.t_step
  gwps = [results.state_metrics['gwp'][i][:results.last_valid_indices[i]] for i in range(results.n_trials)]

  ns = list(range(1, 20))
  ms = list(range(1, 20))
  table = []
  for n in ns:
    row = []
    for m in ms:
      p = np.sum([n_year_doubling_before_m_year_doubling(gwp, t_step, n, m) for gwp in gwps])/results.n_trials if (n > m) else np.nan
      row.append(p)
    table.append(row)

  df = pd.DataFrame(table)
  df.index = ns
  df.columns = ms

  return df

def write_mc_analysis_report(n_trials=100, max_retries=100, include_sample_table=False, report_file_path=None, report_dir_path=None, report=None, output_results_filename=None, input_results_filename=None):
  if report_file_path is None:
    report_file_path = 'mc_analysis.html'

  if input_results_filename:
    with open(input_results_filename, 'rb') as f:
      results = pickle.load(f)
  else:
    results = mc_analysis(n_trials, max_retries)

  if output_results_filename:
    with open(output_results_filename, 'wb') as f:
      pickle.dump(results, f)

  log.info('Writing report...')
  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  description = 'Probability of a full 4 year doubling of GWP before a 1 year doubling of GWP starts'
  report.add_paragraph(f"<span style='font-weight:bold'>Probability of slow takeoff</span>{report.generate_tooltip_html(description)}<span style='font-weight:bold'>:</span> {results.slow_takeoff_count/results.n_trials:.0%}")

  metrics_quantiles = pd.DataFrame(results.metrics_quantiles)
  report.add_data_frame(metrics_quantiles)

  # Plot trajectories
  for state_metric in results.state_metrics:
    results.state_metrics[state_metric] = np.stack(results.state_metrics[state_metric])
    plot_quantiles(results.timesteps, results.state_metrics[state_metric], "Year", state_metric)
    report.add_figure()

  # Add violin plots of scalar metrics
  metric_id_to_human = get_metric_names()

  absolute_violin_metrics = {metric_id_to_human[id]: results.scalar_metrics[id] for id in ['rampup_start', 'agi_year']}
  relative_violin_metrics = {metric_id_to_human[id]: results.scalar_metrics[id] for id in ['billion_agis', 'full_automation']}

  fig, ax = plt.subplots(1,2, num=1, figsize=(10, 6), dpi=80)

  ax[0].set_ylabel('Year')
  sns.violinplot(data = pd.DataFrame(absolute_violin_metrics), ax = ax[0])

  ax[1].set_ylabel('Years')
  sns.violinplot(data = pd.DataFrame(relative_violin_metrics), ax = ax[1], palette = sns.color_palette()[2:])

  plt.subplots_adjust(wspace = 0.3) # Increase spacing between subplots

  report.add_figure()

  # Display input parameter statistics
  param_stats = []
  for key, samples in results.param_samples.iteritems():
    samples = samples.to_numpy()
    stats = [np.mean(samples)] + [np.quantile(samples, q) for q in results.quantiles]
    param_stats.append(stats)
  param_names = results.param_samples.columns
  columns = [['mean'] + results.quantiles]

  # Write down the parameters
  report.add_header("Inputs", level = 3)

  report.add_paragraph(f"<span style='font-weight:bold'>Number of samples:</span> {n_trials}")

  report.add_paragraph("<span style='font-weight:bold'>Rank correlations:</span> <span data-modal-trigger='rank-correlations-modal'><i>click here to view</i>.</span>")
  report.add_data_frame_modal(results.rank_correlations.fillna(''), 'rank-correlations-modal')

  params_stats_table = pd.DataFrame(param_stats, index = param_names, columns = columns)
  params_stats_table.columns.name = 'quantiles'

  report.add_paragraph("<span style='font-weight:bold'>Input statistics:</span> <span data-modal-trigger='input-stats-modal'><i>click here to view</i>.</span>")
  report.add_data_frame_modal(params_stats_table, 'input-stats-modal')

  report.add_data_frame_modal(results.ajeya_cdf, 'ajeya-modal', show_index = False)

  # the parameter full_automation_requirements_training is special (we are sampling from Ajeya's distribution)
  table = report.add_data_frame(results.parameter_table.drop(index = 'full_automation_requirements_training', columns = 'Type'))
  tbody = None
  for element in table.iter():
    if element.tag == 'tbody':
      tbody = element
      break
  tbody.insert(0, et.fromstring(f'''
    <tr>
      <th data-param-id='full_automation_requirements_training'>{get_param_names()['full_automation_requirements_training'] if get_option('human_names') else 'full_automation_requirements_training'}</th>
      <td colspan="4" style="text-align: center">sampled from a clipped Cotra's distribution <span data-modal-trigger="ajeya-modal">(<i>click here to view</i>)</span></td>
    </tr>
  '''))

  if include_sample_table:
    report.add_header("Parameter samples", level = 3)
    report.add_data_frame(results.param_samples)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

# https://stackoverflow.com/questions/18313322/plotting-quantiles-median-and-spread-using-scipy-and-matplotlib
def plot_quantiles(ts, data, xlabel, ylabel, n_quantiles = 7, colormap = cm.Blues):
  # Fix overflows
  UPPER_BOUND = np.quantile(data, 0.95)
  data[data == 0.] = UPPER_BOUND
  data[data > UPPER_BOUND] = UPPER_BOUND

  # Compute quantiles
  n = len(ts)
  percentiles = np.linspace(0,100,n_quantiles)

  marks=np.zeros((n,n_quantiles))
  for i in range(n_quantiles):
    for t in range(n):
      marks[t,i]=np.percentile(data[:,t],percentiles[i])

  # Plot
  half = int((n_quantiles-1)/2)
  fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
  for i in range(half):
    credence = percentiles[(n_quantiles-1)-i] - percentiles[i]
    label = f'{round(credence)}% credence interval'
    ax1.fill_between(ts, marks[:,i],marks[:,-(i+1)],color=colormap(i/half), label=label)
  ax1.plot(ts, marks[:,half],color='k', label="median")

  # Sort the legend
  legend_handles, legend_labels = ax1.get_legend_handles_labels()
  legend_handles.reverse()
  legend_labels.reverse()

  ax1.set_title("Takeoff simulation", fontsize=15)
  ax1.set_yscale("log")
  ax1.tick_params(labelsize=11.5)
  ax1.set_xlabel(xlabel, fontsize=14)
  ax1.set_ylabel(ylabel, fontsize=14)
  fig.tight_layout()
  plt.legend(legend_handles, legend_labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# -----------------------------------------------------------------------------
# Distributions
# -----------------------------------------------------------------------------

class ParamsDistribution():
  """ Joint parameter distribution. """

  def __init__(self, ensure_no_automatable_goods_tasks = True):
    """
      If ensure_no_automatable_goods_tasks is True, we'll make sure none of the samples
      represent an scenario in which there is some "goods" task initially automatable.
    """

    # Retrieve parameter table
    log.info('Retrieving parameters...')
    parameter_table = get_parameter_table()
    parameter_table = parameter_table[['Conservative', 'Best guess', 'Aggressive', 'Type']]

    # Disable the runtime-training tradeoff for the MC analysis
    parameter_table.at['runtime_training_tradeoff', 'Conservative'] = None
    parameter_table.at['runtime_training_tradeoff', 'Best guess']   = 0
    parameter_table.at['runtime_training_tradeoff', 'Aggressive']   = None

    parameter_table.at['runtime_training_max_tradeoff', 'Conservative'] = None
    parameter_table.at['runtime_training_max_tradeoff', 'Best guess']   = 1
    parameter_table.at['runtime_training_max_tradeoff', 'Aggressive']   = None

    rank_correlations = get_rank_correlations()

    marginals = {}
    directions = {}
    for parameter, row in parameter_table.iterrows():
      if not np.isnan(row['Conservative']) and not np.isnan(row['Aggressive']):
        directions[parameter] = +1 if (row['Conservative'] < row['Aggressive']) else -1
        marginal = SkewedLogUniform(
          row['Conservative'],
          row['Best guess'],
          row['Aggressive'],
          kind = row['Type']
        )
      else:
        marginal = PointDistribution(row['Best guess'])
      marginals[parameter] = marginal

    lower_ajeya_bound = \
        (marginals['initial_biggest_training_run'].b * marginals['runtime_training_max_tradeoff'].b) \
        * marginals['flop_gap_training'].a**(10.5/7)

    marginals['full_automation_requirements_training'] = \
        AjeyaDistribution(lower_bound = lower_ajeya_bound)

    pairwise_rank_corr = {}
    for left in marginals.keys():
      for right in marginals.keys():
        if right not in rank_correlations or left not in rank_correlations:
          continue

        if isinstance(marginals[right], PointDistribution) or isinstance(marginals[left], PointDistribution):
          continue

        r = rank_correlations[right][left]
        if not np.isnan(r) and r != 0:
          pairwise_rank_corr[(left, right)] = r * directions[left]*directions[right]

    self.marginals = marginals
    self.pairwise_rank_corr = pairwise_rank_corr
    self.parameter_table = parameter_table
    self.rank_correlations = rank_correlations
    self.joint_dist = joint_distribution.JointDistribution(marginals, pairwise_rank_corr, rank_corr_method = "spearman")
    self.ensure_no_automatable_goods_tasks = ensure_no_automatable_goods_tasks

  def rvs(self, count):
    # statsmodels.distributions.copula.copulas throws an exception when we ask less than 2 samples from it.
    # We could make this more efficient, but it's probably not worth it.
    actual_count = max(count, 2)
    samples = self.joint_dist.rvs(actual_count)[:count]

    if self.ensure_no_automatable_goods_tasks:
      # Resample the training gap to ensure no tasks is automatable from the beginning
      gap_marginal = self.marginals['flop_gap_training']
      for i, row in samples.iterrows():
        max_gap = (row['full_automation_requirements_training']/(row['initial_biggest_training_run'] * row['runtime_training_max_tradeoff']))**(7/10.5)
        gap_marginal.set_upper_bound(max_gap)
        samples.at[i, 'flop_gap_training'] = gap_marginal.rvs()
      gap_marginal.set_upper_bound(None)

    return samples

  def get_marginals(self):
    return self.marginals

  def get_rank_correlations(self):
    return self.pairwise_rank_corr

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

  def _ppf(self, q):
    upper_q = self._cdf(self.upper_bound)
    q = q * upper_q
    return super()._ppf(q)

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
      cd = 0
    elif s*y < s*self.med:
      cd = 1./2 * (y - self.low) / (self.med - self.low)
    elif s*y < s*self.high:
      cd = 1./2 + 1./2 * (y - self.med) / (self.high - self.med)
    else:
      cd = 1

    return cd

class PointDistribution(rv_continuous):
  def __init__(self, v):
    self.v = v
    super().__init__(a = v, b = v)

  def _ppf(self, q):
    return self.v

  def get_value(self):
    return self.v

if __name__ == '__main__':
  parser = init_cli_arguments()

  parser.add_argument(
    "-n",
    "--n-trials",
    type=int,
    default=100,
  )

  parser.add_argument(
    "-r",
    "--max-retries",
    type=int,
    default=100,
  )

  parser.add_argument(
    "--include-sample-table",
    action='store_true',
  )

  parser.add_argument(
    "--input-results-file",
    help = 'Read the MC results from this file (pickle) instead of regenerating them',
  )

  parser.add_argument(
    "--output-results-file",
    help = 'Store the results of the analysis in this file (pickle)',
  )

  args = handle_cli_arguments(parser)

  write_mc_analysis_report(
    n_trials=args.n_trials,
    max_retries=args.max_retries,
    include_sample_table=args.include_sample_table,
    report_file_path=args.output_file,
    report_dir_path=args.output_dir,
    output_results_filename=args.output_results_file,
    input_results_filename=args.input_results_file,
  )

