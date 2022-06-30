"""
Monte carlo analysis.
"""

from . import log
from . import *

from scipy.special import erfinv
from numpy.random import default_rng
rng = default_rng()
from matplotlib import cm

def credence_interval(low, med, high, alpha=0.9, kind='pos', skewed_sampling=True):
  """ Returns a random number sampled from a lognormal
      with a given 90% confidence interval
      If frac, we transform the space to sample from odds space
  """

  if high < low:
    low, high = high, low

  if kind == "frac":
    low = low / (1. - low)
    med = med / (1. - med)
    high = high / (1. - high)

  elif kind == 'neg':
    low, med, high = -high, -med, -low

  elif kind == "inv_frac":
    low, med, high = 1/high, 1/med, 1/low
    low = low / (1. - low)
    med = med / (1. - med)
    high = high / (1. - high)

  elif not kind == "pos":
    raise ValueError(f"Unimplemented kind: {kind}")

  assert low < med and med < high
  
  if not skewed_sampling:
    inv_error = erfinv(alpha)
    mu = np.log(np.sqrt(low* high))
    sigma = (1./(np.sqrt(2)*inv_error))*np.log(np.sqrt(high/low))
    sample = rng.lognormal(mu, sigma)
  
  else: #if skewed_sampling:
    coin_flip = rng.integers(0,1)
    if coin_flip == 0:
      low = med
    else: # if coin_flip == 1:
      high = med
    
    sample = np.exp(rng.uniform(np.log(low), np.log(high)))

  if kind == "frac":
    sample = 1. / (1. + 1. / sample)
  elif kind == "neg":
    sample = -sample
  elif kind == "inv_frac":
    sample = 1. + 1. / sample

  return sample

def sample_from_cdf(cdf):
  """cdf must be a Numpy table with the values in the first column and the probabilities in the second"""
  p = np.random.random()
  for row in cdf:
    if row[1] > p:
      break
  sample = row[0]
  return sample

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

def mc_analysis(n_trials=100, report_file_path=None, report_dir_path=None):
  if report_file_path is None:
    report_file_path = 'mc_analysis.html'

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

  # Retrieve parameter table
  log.info('Retrieving parameters...')
  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")

  # We are sampling full_automation_requirements_training from Ajeya's distribution
  ajeya_cdf = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=1177136586')
  ajeya_cdf_numpy = ajeya_cdf.to_numpy()
  parameter_table.drop('full_automation_requirements_training', inplace = True)

  param_samples = {
    parameter : []
    for parameter, row in parameter_table.iterrows()
    if not np.isnan(row['Conservative']) and not np.isnan(row['Aggressive'])
  }
  param_samples['full_automation_requirements_training'] = []

  log.info(f'Running simulations...')
  for trial in range(n_trials):
    log.info(f'  Running simulation {trial+1}/{n_trials}...')
    # Sample parameters
    mc_params = {
        parameter : credence_interval(row['Conservative'],
                                      row['Best guess'],
                                      row['Aggressive'],
                                      kind=row['Type'])
                    if not np.isnan(row['Conservative'])\
                    and not np.isnan(row['Aggressive'])
                    else row['Best guess']
        for parameter, row in parameter_table.iterrows()
    }

    mc_params['full_automation_requirements_training'] = 10**sample_from_cdf(ajeya_cdf_numpy)

    # Collect parameter samples
    for param, samples in param_samples.items():
      samples.append(mc_params[param])

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
      # if metric_value < 0.:
      #   print(f"{scalar_metric} is negative!")
      #   raise Exception
      scalar_metrics[scalar_metric].append(metric_value)

    for state_metric in state_metrics:
      metric_value = getattr(mc_model, state_metric)
      assert metric_value.shape == (mc_model.n_timesteps,)
      state_metrics[state_metric].append(metric_value)

  log.info('Writing report...')
  report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)


  # Display summary of scalar metrics
  quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
  results = []
  for q in quantiles:
    result = {"quantile" : q}
    for scalar_metric in scalar_metrics:
      result[scalar_metric] = np.quantile(scalar_metrics[scalar_metric],q)
    results.append(result)
  
  ## Add mean
  result = {"quantile" : "mean"}
  for scalar_metric in scalar_metrics:
    result[scalar_metric] = np.mean(scalar_metrics[scalar_metric])
  results.append(result)
  
  results = pd.DataFrame(results)
  display(results)
  report.add_data_frame(results)

  # Plot trajectories
  for state_metric in state_metrics:
    state_metrics[state_metric] = np.stack(state_metrics[state_metric])
    plot_quantiles(mc_model.timesteps, state_metrics[state_metric], "Year", state_metric)
    report.add_figure()

  # Display input parameter statistics
  param_stats = []
  for samples in param_samples.values():
    stats = [np.mean(samples)] + [np.quantile(samples, q) for q in quantiles]
    param_stats.append(stats)
  param_names = param_samples.keys()
  columns = [['mean'] + quantiles]

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
  report.add_data_frame_modal(ajeya_cdf, 'ajeya-modal', show_index = False)
  report.add_data_frame(parameter_table)

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-n",
    "--n-trials",
    type=int,
    default=100,
  )
  args = parser.parse_args()
  mc_analysis(n_trials=args.n_trials, report_file_path=args.output_file, report_dir_path=args.output_dir)

