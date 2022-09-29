"""
Monte carlo analysis.
"""

from . import log
from . import *

import json
import pickle
import traceback
import seaborn as sns
from scipy.stats import rv_continuous
from scipy.special import erfinv
from numpy.random import default_rng
from matplotlib import cm
from xml.etree import ElementTree as et
from ..core.utils import get_clipped_ajeya_dist, get_param_names, get_metric_names
from ..stats.distributions import *
from statsmodels.distributions.empirical_distribution import ECDF

rng = default_rng()

class McAnalysisResults:
  pass

class TooManyRetries(Exception):
  pass

def mc_analysis(n_trials = 100, max_retries = 100):
  scalar_metrics = {}

  for takeoff_metric in SimulateTakeOff.takeoff_metrics:
    scalar_metrics[takeoff_metric] = []

  for m in ['rampup_start', 'agi_year']:
    scalar_metrics[m] = []

  state_metrics = {
      metric : [] for metric in ['biggest_training_run', 'gwp']
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
    row = {"Quantile" : q}
    for scalar_metric in scalar_metrics:
      row[scalar_metric] = np.quantile(scalar_metrics[scalar_metric], q)
    metrics_quantiles.append(row)

  ## Add mean
  row = {"Quantile" : "mean"}
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
  results.t_start            = mc_model.t_start
  results.t_end              = mc_model.t_end
  results.param_samples      = pd.concat(samples, ignore_index = True)
  results.parameter_table    = params_dist.parameter_table
  results.rank_correlations  = params_dist.rank_correlations
  results.slow_takeoff_count = slow_takeoff_count
  results.last_valid_indices = last_valid_indices

  reqs_marginal = params_dist.marginals['full_automation_requirements_training']
  results.ajeya_cdf = reqs_marginal.cdf_pd if isinstance(reqs_marginal, AjeyaDistribution) else None

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

def write_mc_analysis_report(
    n_trials=100, max_retries=100, include_sample_table=False, report_file_path=None,
    report_dir_path=None, report=None, output_results_filename=None, input_results_filename=None,
    results=None
  ):

  if report_file_path is None:
    report_file_path = 'mc_analysis.html'

  if not results:
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

  #
  # Add a mini-widget in a tooltip to let the user select the definition of "slow takeoff"
  #

  # Create the table
  gwps = [results.state_metrics['gwp'][i][:results.last_valid_indices[i]] for i in range(results.n_trials)]
  takeoff_probability_table = []
  for n in range(1, 20):
    row = []
    for m in range(1, 20):
      p = np.sum([n_year_doubling_before_m_year_doubling(gwp, results.t_step, n, m) for gwp in gwps])/results.n_trials if (n > m) else np.nan
      row.append(p)
    takeoff_probability_table.append(row)

  # Add the tooltip
  description = '''Probability of a full <input class='doubling-years-inputs' id='doubling-years-input-m' value='4' style='display:inline-block'> year doubling of GWP before a <input class='doubling-years-inputs' id='doubling-years-input-n' value='1'> year doubling of GWP starts'''
  report.add_paragraph(f"<span style='font-weight:bold'>Probability of slow takeoff</span>{report.generate_tooltip_html(description, on_mount = 'initialize_takeoff_probability_mini_widget()', triggers = 'mouseenter click', classes = 'slow-takeoff-probability-tooltip-info')}<span style='font-weight:bold'>:</span> <span id='slow-takeoff-probability'>{results.slow_takeoff_count/results.n_trials:.0%}</span>")

  # Style
  report.head.append(et.fromstring('''
    <style>
      .doubling-years-inputs {
        width: 2em;
        text-align: center;
        border: none;
        border-bottom: 1px dashed black;
      }

      .slow-takeoff-probability-tooltip-info {
        cursor: pointer;
      }
    </style>
  '''))

  # JS
  report.body.append(et.fromstring('<script>' + Report.escape('''
      let takeoff_probability_mini_widget_initialized = false;

      function initialize_takeoff_probability_mini_widget() {
        if (takeoff_probability_mini_widget_initialized) {
          return;
        }

        // p(full <row + 1> year doubling before the start of a <col + 1> year doubling)
        let p_table = ''' + json.dumps(takeoff_probability_table) + ''';

        let n_input = document.getElementById('doubling-years-input-n');
        let m_input = document.getElementById('doubling-years-input-m');
        let probability = document.getElementById('slow-takeoff-probability');

        function update_slow_takeoff_probability() {
          let m = parseInt(m_input.value);
          let n = parseInt(n_input.value);
          let p = NaN;

          if ((1 <= m && m <= p_table.length) && (1 <= n && n <= p_table[0].length)) {
            p = p_table[m-1][n-1] * 100;
          }

          probability.innerHTML = Number.isNaN(p) ? '--' : `${p.toFixed()}%`;
        }

        m_input.addEventListener('input', update_slow_takeoff_probability);
        n_input.addEventListener('input', update_slow_takeoff_probability);

        takeoff_probability_mini_widget_initialized = true;
      }
  ''') + '</script>'))

  # Metrics quantiles table
  metrics_quantiles = pd.DataFrame(results.metrics_quantiles)
  metrics_quantiles_styled = metrics_quantiles.style.format(lambda x: x if isinstance(x, str) else f'{x:.2f}').hide(axis = 'index')
  report.add_data_frame(metrics_quantiles_styled, show_importance_selector = True,
      important_columns_to_keep = [0], label = 'metrics', show_index = False,
  )

  # Plot CDFs of scalar metrics
  metric_id_to_human = get_metric_names()

  def plot_ecdf(x, limit, label, color = None, normalize = False):
    x = np.sort(x)

    ecdf = ECDF(x)(x)

    ecdf = np.insert(ecdf, 0, 0)
    x = np.insert(x, 0, x[0])

    ecdf = ecdf[x < limit]
    x = x[x < limit]

    ecdf = np.append(ecdf, ecdf[-1])
    x = np.append(x, limit)

    if normalize:
      ecdf /= ecdf[-1]

    plt.step(x, ecdf, where = 'post', label = label, color = color)
    plt.ylim(0, 1)

  sns.reset_defaults()

  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  color_index = 0

  # Relative metrics
  plt.figure(figsize=(10,6))
  for metric in ['full_automation', 'billion_agis']:
    plot_ecdf(results.scalar_metrics[metric], limit = results.t_end - results.t_start, label = metric_id_to_human[metric], color = colors[color_index], normalize = True)
    color_index += 1
  text = f'p(takeoff before year {results.t_end}) = {np.mean(np.array(results.scalar_metrics["billion_agis"]) < results.t_end - results.t_start):.0%}'
  plt.text(results.t_end - results.t_start - 2, 0.95, text, va = 'top', ha = 'right')
  plt.xlabel('Years')
  plt.ylabel(f'CDF\n(conditional on takeoff happening before {results.t_end})')
  plt.gca().legend(loc = 'lower right')

  report.add_figure()

  # Absolute metrics
  plt.figure(figsize=(10,6))
  for metric in ['rampup_start', 'agi_year']:
    plot_ecdf(results.scalar_metrics[metric], limit = results.t_end, label = metric_id_to_human[metric], color = colors[color_index])
    color_index += 1
  plt.ylabel('CDF')
  plt.xlabel('Year')
  plt.gca().legend(loc = 'lower right')

  report.add_figure()

  # Plot trajectories
  metrics = {'biggest_training_run': 'Biggest training run'}
  for metric in metrics:
    results.state_metrics[metric] = np.stack(results.state_metrics[metric])
    plot_quantiles(results.timesteps, results.state_metrics[metric], "Year", metrics[metric])
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

  report.add_paragraph(f"<span style='font-weight:bold'>Number of samples:</span> {results.n_trials}")

  report.add_paragraph("<span style='font-weight:bold'>Rank correlations:</span> <span data-modal-trigger='rank-correlations-modal'><i>click here to view</i>.</span>")
  report.add_data_frame_modal(results.rank_correlations.fillna(''), 'rank-correlations-modal')

  params_stats_table = pd.DataFrame(param_stats, index = param_names, columns = columns)
  params_stats_table.columns.name = 'quantiles'

  report.add_paragraph("<span style='font-weight:bold'>Input statistics:</span> <span data-modal-trigger='input-stats-modal'><i>click here to view</i>.</span>")
  report.add_data_frame_modal(params_stats_table, 'input-stats-modal')

  if results.ajeya_cdf is not None:
    report.add_data_frame_modal(results.ajeya_cdf, 'ajeya-modal', show_index = False)

    # the parameter full_automation_requirements_training is special (we might be sampling from Ajeya's distribution)
    inputs_table = report.add_data_frame(results.parameter_table.drop(index = 'full_automation_requirements_training', columns = 'Type'), show_justifications = True)
    tbody = None
    for element in inputs_table.iter():
      if element.tag == 'tbody':
        tbody = element
        break
    tbody.insert(0, et.fromstring(f'''
      <tr>
        <th data-param-id='full_automation_requirements_training'>{get_param_names()['full_automation_requirements_training'] if get_option('human_names') else 'full_automation_requirements_training'}</th>
        <td colspan="4" style="text-align: center">sampled from a clipped Cotra's distribution <span data-modal-trigger="ajeya-modal">(<i>click here to view</i>)</span></td>
      </tr>
    '''))
  else:
    inputs_table = report.add_data_frame(results.parameter_table.drop(columns = 'Type'), show_justifications = True)

  report.add_importance_selector(inputs_table, label = 'parameters', layout = 'vertical')

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
  plt.figure(figsize=(10,6))
  for i in range(half):
    credence = percentiles[(n_quantiles-1)-i] - percentiles[i]
    label = f'{round(credence)}% credence interval'
    plt.gca().fill_between(ts, marks[:,i],marks[:,-(i+1)],color=colormap(i/half), label=label)
  plt.gca().plot(ts, marks[:,half],color='k', label="median")

  # Sort the legend
  legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
  legend_handles.reverse()
  legend_labels.reverse()

  plt.gca().set_yscale("log")
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.legend(legend_handles, legend_labels, loc='upper left')

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

