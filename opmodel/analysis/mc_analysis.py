"""
Monte carlo analysis.
"""

from . import log
from . import *

from scipy.stats import rv_continuous
from scipy.special import erfinv
from numpy.random import default_rng
from matplotlib import cm
from copula_wrapper import joint_distribution

rng = default_rng()

class McAnalysisResults:
  pass

def mc_analysis(n_trials = 100):
  scalar_metrics = ['rampup_start', 'agi_year']
  state_metrics = ['gwp', 'biggest_training_run', 'compute']

  scalar_metrics = {
      scalar_metric : [] for scalar_metric in scalar_metrics
  }
  for takeoff_metric in SimulateTakeOff.takeoff_metrics:
    scalar_metrics[takeoff_metric] = []

  state_metrics = {
      state_metric : [] for state_metric in state_metrics
  }

  samples, parameter_table, rank_correlations = sample_params(n_trials)

  log.info(f'Running simulations...')
  for trial in range(n_trials):
    log.info(f'  Running simulation {trial+1}/{n_trials}...')

    mc_params = samples.iloc[trial].to_dict()

    # Run simulation
    mc_model = SimulateTakeOff(**mc_params)
    mc_model.run_simulation()

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
  results.quantiles         = quantiles
  results.metrics_quantiles = metrics_quantiles
  results.state_metrics     = state_metrics
  results.n_trials          = n_trials
  results.timesteps         = mc_model.timesteps
  results.param_samples     = samples
  results.ajeya_cdf         = AjeyaDistribution.cdf_pd
  results.parameter_table   = parameter_table

  return results

def sample_params(param_count):
  # Retrieve parameter table
  log.info('Retrieving parameters...')
  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")

  rank_correlations = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=605978895', skiprows = 2)
  rank_correlations = rank_correlations.set_index(rank_correlations.columns[0])

  # We'll use Ajeya's distribution for this one
  parameter_table.drop('full_automation_requirements_training', inplace = True)

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

  marginals['full_automation_requirements_training'] = AjeyaDistribution()

  pairwise_rank_corr = {}
  for left in marginals:
    for right in marginals:
      r = rank_correlations[right][left]
      if not np.isnan(r) and r != 0:
        pairwise_rank_corr[(left, right)] = r

  log.info(f'Generating samples...')
  joint = joint_distribution.JointDistribution(marginals, pairwise_rank_corr, rank_corr_method = "spearman")
  samples = joint.rvs(param_count)

  return samples, parameter_table, rank_correlations

  
def write_mc_analysis_report(n_trials=100, report_file_path=None, report_dir_path=None):
  if report_file_path is None:
    report_file_path = 'mc_analysis.html'

  results = mc_analysis(n_trials)

  log.info('Writing report...')
  report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  metrics_quantiles = pd.DataFrame(results.metrics_quantiles)
  display(metrics_quantiles)
  report.add_data_frame(metrics_quantiles)

  # Plot trajectories
  for state_metric in results.state_metrics:
    results.state_metrics[state_metric] = np.stack(results.state_metrics[state_metric])
    plot_quantiles(results.timesteps, results.state_metrics[state_metric], "Year", state_metric)
    report.add_figure()

  # Display input parameter statistics
  param_stats = []
  for key, samples in results.param_samples.iteritems():
    samples = samples.to_numpy()
    stats = [np.mean(samples)] + [np.quantile(samples, q) for q in results.quantiles]
    param_stats.append(stats)
  param_names = results.param_samples.columns
  columns = [['mean'] + results.quantiles]

  report.add_header("Input parameters stats", level = 3)
  params_stats_table = pd.DataFrame(param_stats, index = param_names, columns = columns)
  params_stats_table.columns.name = 'quantiles'
  report.add_data_frame(params_stats_table)

  # Write down the parameters
  report.add_header("Inputs", level = 3)
  report.add_paragraph(f"<span style='font-weight:bold'>n_trials:</span> {n_trials}")
  report.add_paragraph(f"""
    <span style='font-weight:bold'>full_automation_requirements_training:</span>
    <span data-modal-trigger="ajeya-modal">sampled from Ajeya's distribution (click to view)</span>
  """)
  report.add_data_frame_modal(results.ajeya_cdf, 'ajeya-modal', show_index = False)
  report.add_data_frame(results.parameter_table)

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
  ax1.plot(ts, marks[:,half],color='k')
  for i in range(half):
    ax1.fill_between(ts, marks[:,i],marks[:,-(i+1)],color=colormap(i/half))

  ax1.set_title("Takeoff simulation", fontsize=15)
  ax1.set_yscale("log")
  ax1.tick_params(labelsize=11.5)
  ax1.set_xlabel(xlabel, fontsize=14)
  ax1.set_ylabel(ylabel, fontsize=14)
  fig.tight_layout()


# -----------------------------------------------------------------------------
# Distributions
# -----------------------------------------------------------------------------

class AjeyaDistribution(rv_continuous):
  cdf_pd = None
  cdf_np = None

  def __init__(self):
    if AjeyaDistribution.cdf_np is None:
      AjeyaDistribution.cdf_pd = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=1177136586')
      AjeyaDistribution.cdf_np = AjeyaDistribution.cdf_pd.to_numpy()
    ajeya_cdf_log10 = AjeyaDistribution.cdf_np

    self.ajeya_cdf_log10 = ajeya_cdf_log10
    self.v = ajeya_cdf_log10[:, 0]
    self.p = ajeya_cdf_log10[:, 1]

    # Normalize the distribution
    self.p /= self.p[-1]

    super().__init__(a = 10**np.min(self.v), b = 10**np.max(self.v))

  def _cdf(self, x):
    y = np.log10(x)
    v = np.interp(y, self.v, self.p)
    return v

  def _ppf(self, q):
    # We'll approximate the PPF ourselves (scipy is unable to handle it)
    v = np.interp(q, self.p, self.v)
    return 10**v

class SkewedLogUniform(rv_continuous):
  def __init__(self, low, med, high, kind = 'pos'):
    if high < low:
      low, high = high, low

    super().__init__(a = low, b = high)

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


if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-n",
    "--n-trials",
    type=int,
    default=100,
  )
  args = parser.parse_args()
  write_mc_analysis_report(n_trials=args.n_trials, report_file_path=args.output_file, report_dir_path=args.output_dir)

