"""
Monte carlo analysis.
"""

from . import log
from . import *

import json
import dill as pickle
import traceback
import seaborn as sns
import colorsys
import matplotlib.colors as mc
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import rv_continuous, gaussian_kde
from scipy.interpolate import interp1d
from scipy.special import erfinv
from numpy.random import default_rng
from matplotlib import cm
from xml.etree import ElementTree as et
from ..core.utils import get_clipped_ajeya_dist, get_param_names, get_metric_names, get_most_important_metrics, pluralize
from ..stats.distributions import *
from statsmodels.distributions.empirical_distribution import ECDF

rng = default_rng()

class McAnalysisResults:
  pass

class TooManyRetries(Exception):
  pass

class YearsBeforeAgiMetric:
  """ Metrics for the "years before AGI" quantiles table """

  # This is hacky
  def __init__(self, name, get_values, interpolation = 'log', format_quantile = lambda x: str(x), remarks = None):
    self.name = name
    self.get_values = get_values
    self.interpolation = interpolation
    self.format_quantile = format_quantile
    self.remarks = remarks

  def get_value_at_year(self, year, model, no_automation_model):
    ts, values = self.get_values(model)
    no_auto_ts, no_auto_vals = self.get_values(no_automation_model)

    interpolator = interp1d(
      ts,
      values if (self.interpolation == 'linear') else np.log10(values),
      fill_value = 'extrapolate'
    )

    no_auto_interpolator = interp1d(
      no_auto_ts,
      no_auto_vals if (self.interpolation == 'linear') else np.log10(no_auto_vals),
      fill_value = 'extrapolate'
    )

    if year < model.t_start:
      # Extrapolate
      value = no_auto_interpolator(year)
    else:
      value = interpolator(year)

    if self.interpolation == 'log': value = 10**value

    return value

def mc_analysis(n_trials = 100, max_retries = 100):
  scalar_metrics = {}

  for takeoff_metric in SimulateTakeOff.takeoff_metrics:
    scalar_metrics[takeoff_metric] = []

  for m in ['rampup_start', 'agi_year']:
    scalar_metrics[m] = []

  state_metrics = {
      metric : [] for metric in ['biggest_training_run', 'gwp']
  }

  # Metrics for the "years before AGI" tables

  years_before_agi = [0, 1, 2, 5, 10]
  metrics_before_agi = []

  # Doubling times metrics
  class DoublingYearsBeforeAgiMetric(YearsBeforeAgiMetric):
    def __init__(self, attr, remarks = None):
      self.name = f'{attr} doubling time (years)'
      self.format_quantile = lambda x: f'{x:.1f}' if x <= 100 else 'N/A'
      self.attr = attr
      self.remarks = remarks

    def get_value_at_year(self, year, normal_model, no_automation_model):
      model = no_automation_model if (year < normal_model.t_start) else normal_model

      ts = model.timesteps
      vals = getattr(model, self.attr)
      interpolator = interp1d(ts, np.log10(vals), fill_value = 'extrapolate')

      denominator = np.log2(10**interpolator(year+1)/10**interpolator(year))
      doubling_time = 1/denominator if (denominator != 0) else 1e10

      return doubling_time

  def add_doubling_time_metric(metric, remarks = None):
    metrics_before_agi.append(DoublingYearsBeforeAgiMetric(metric, remarks))

  add_doubling_time_metric('gwp')
  add_doubling_time_metric('hardware_performance')
  add_doubling_time_metric('software', 'N/A: at physical limit')

  # Rest of the metric
  def add_model_var_metric(var, format_quantile):
    metrics_before_agi.append(
      YearsBeforeAgiMetric(
        f'{var}',
        lambda model, _var=var: (model.timesteps, getattr(model, _var)),
        interpolation = 'log',
        format_quantile = format_quantile,
      )
    )
  add_model_var_metric('biggest_training_run', format_quantile = lambda x: f'{x:.1e}')
  add_model_var_metric('frac_compute_training', format_quantile = lambda x: f'{x:.2%}')
  add_model_var_metric('frac_gwp_compute', format_quantile = lambda x: f'{x:.2%}')

  metrics_before_agi_values = {
      metric: {year: [] for year in years_before_agi} for metric in metrics_before_agi
  }

  slow_takeoff_count = 0

  parameter_table = get_parameter_table()
  parameter_table = parameter_table[['Conservative', 'Best guess', 'Aggressive', 'Type']]

  parameter_table.at['runtime_training_tradeoff', 'Conservative'] = 3
  parameter_table.at['runtime_training_tradeoff', 'Best guess']   = 1.5
  parameter_table.at['runtime_training_tradeoff', 'Aggressive']   = 0.5

  parameter_table.at['runtime_training_max_tradeoff', 'Conservative'] = 10**0.001
  parameter_table.at['runtime_training_max_tradeoff', 'Best guess']   = 10**1.5
  parameter_table.at['runtime_training_max_tradeoff', 'Aggressive']   = 10**3

  params_dist = TakeoffParamsDist(parameter_table=parameter_table)
  samples = []

  last_valid_indices = []

  t_start = get_option('t_start', 2022)
  t_end   = get_option('t_end',   2100)
  t_step  = get_option('t_step',  0.1)
  timesteps = np.arange(t_start, t_end, t_step)

  log.info(f'Running simulations...')
  log.indent()
  for trial in range(n_trials):
    for i in range(max_retries):
      # Try to run the simulation
      try:
        log.info(f'Running simulation {trial+1}/{n_trials}...')

        sample = params_dist.rvs(1)
        mc_params = {param: sample[param][0] for param in sample}

        model = SimulateTakeOff(**mc_params, t_start = t_start, t_end_min = t_end, compute_shares = False)

        # Check that no goods task is automatable from the beginning (except for the first one)
        runtime_training_max_tradeoff = model.runtime_training_max_tradeoff if model.runtime_training_tradeoff is not None else 1.
        assert(not np.any(model.automation_training_flops_goods[1:] < model.initial_biggest_training_run * runtime_training_max_tradeoff))

        model.run_simulation()
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
        metric_value = model.takeoff_metrics[scalar_metric]
        assert (np.isnan(metric_value) or metric_value >= 0), f"{scalar_metric} is negative!"
      else:
        metric_value = getattr(model, scalar_metric)

      if scalar_metric == "rampup_start" and metric_value is None:
        metric_value = model.t_end
      elif scalar_metric == "agi_year" and metric_value is None:
        metric_value = model.t_end
      scalar_metrics[scalar_metric].append(metric_value)

    for state_metric in state_metrics:
      metric_value = getattr(model, state_metric)
      assert metric_value.shape == (model.n_timesteps,)
      state_metrics[state_metric].append(metric_value)

    if model.agi_year is not None:
      no_automation_mc_params = mc_params.copy()
      no_automation_mc_params['full_automation_requirements_training'] = 1e100
      no_automation_mc_params['flop_gap_training'] = 2
      no_automation_model = SimulateTakeOff(**no_automation_mc_params, t_start = t_start, t_end = t_start + (2 + t_step))
      no_automation_model.run_simulation()

      assert(np.all(no_automation_model.frac_tasks_automated_goods < 1) and np.all(no_automation_model.frac_tasks_automated_rnd < 1))

      for metric in metrics_before_agi:
        for year, year_values in metrics_before_agi_values[metric].items():
          value = metric.get_value_at_year(model.agi_year - year, model, no_automation_model)
          year_values.append(value)

    last_valid_indices.append(model.t_idx)

    if is_slow_takeoff(model):
      slow_takeoff_count += 1

  log.deindent()

  for scalar_metric in scalar_metrics:
    scalar_metrics[scalar_metric] = np.array(scalar_metrics[scalar_metric])

  ## Summaries
  quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

  # Summary of scalar metrics
  metrics_quantiles = []
  for q in quantiles:
    row = {"Quantile" : q}
    for scalar_metric in scalar_metrics:
      value = np.quantile(filter_nans(scalar_metrics[scalar_metric]), q)
      row[scalar_metric] = value if (value < t_end) else f'≥ {t_end}'
    metrics_quantiles.append(row)

  # Summary of the "years before AGI" quantiles
  metrics_before_agi_quantiles = {}
  import dill as pickle
  with open('/home/edu/.tmp/trash/metrics.pickle', 'wb') as f:
    pickle.dump(metrics_before_agi_values, f)
  for metric, metric_samples in metrics_before_agi_values.items():
    table = []
    for q in quantiles:
      row = {"Percentile" : f'{q:.0%}'}
      for year, values in metric_samples.items():
        column_name = 'At time of AGI' if (year == 0) else f'{year} {pluralize("year", year)} before AGI'
        row[column_name] = np.quantile(values, q)
      table.append(row)
    metrics_before_agi_quantiles[metric] = table

  ## Add mean
  row = {"Quantile" : "mean"}
  for scalar_metric in scalar_metrics:
    row[scalar_metric] = np.mean(filter_nans(scalar_metrics[scalar_metric]))
  metrics_quantiles.append(row)

  results = McAnalysisResults()
  results.quantiles                    = quantiles
  results.metrics_quantiles            = metrics_quantiles
  results.state_metrics                = state_metrics
  results.scalar_metrics               = scalar_metrics
  results.n_trials                     = n_trials
  results.timesteps                    = timesteps
  results.t_step                       = t_step
  results.t_start                      = t_start
  results.t_end                        = t_end
  results.param_samples                = pd.concat(samples, ignore_index = True)
  results.parameter_table              = params_dist.parameter_table
  results.rank_correlations            = params_dist.rank_correlations
  results.slow_takeoff_count           = slow_takeoff_count
  results.last_valid_indices           = last_valid_indices
  results.metrics_before_agi_quantiles = metrics_before_agi_quantiles

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
  most_important_metrics = get_most_important_metrics()
  def keep_cell(row, col, index_r, index_c, cell):
    col_condition = (col in [0]) or (index_c in most_important_metrics)
    row_condition = (row in [0]) or (index_r != 'mean')
    return col_condition and row_condition

  def process_header(row, col, index_r, index_c, cell):
    if row == 0:
      if index_c in SimulateTakeOff.takeoff_metrics:
        cell.attrib['data-meaning-suffix'] = f'<br><br><i>Conditional on takeoff happening before {results.t_end}.</i>'

  metrics_quantiles = pd.DataFrame(results.metrics_quantiles)
  metrics_quantiles_styled = metrics_quantiles.style.format(lambda x: x if isinstance(x, str) else f'{x:.2f}').hide(axis = 'index')
  table_container = report.add_data_frame(metrics_quantiles_styled, show_importance_selector = True,
      keep_cell = keep_cell, label = 'metrics', show_index = False,
  )
  report.apply_to_table(table_container, process_header)

  # Plot CDFs of scalar metrics
  metric_id_to_human = get_metric_names()

  def plot_ecdf(x, limits, label, color = None, normalize = False):
    x = filter_nans(x)
    x = np.sort(x)

    ecdf = ECDF(x)(x)

    ecdf = np.insert(ecdf, 0, 0)
    x = np.insert(x, 0, x[0])

    ecdf = ecdf[x < limits[1]]
    x = x[x < limits[1]]

    ecdf = ecdf[x >= limits[0]]
    x = x[x >= limits[0]]

    ecdf = np.append(ecdf, ecdf[-1])
    x = np.append(x, limits[1])

    if normalize:
      ecdf /= ecdf[-1]

    plt.step(x, ecdf, where = 'post', label = label, color = color)
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

  def plot_epdf(x, limits, label, color = None, normalize = False):
    x = filter_nans(x)
    x = x[x < limits[1]]
    x = x[x >= limits[0]]

    t = np.linspace(limits[0], limits[1], 500)
    t0 = t[0] - 0.1

    plt.plot(t, 1/(t - t0) * gaussian_kde(np.log(x - t0))(np.log(t - t0)), label = label, color = color)

    c = colorsys.rgb_to_hls(*mc.to_rgb(color))
    hist_color = colorsys.hls_to_rgb(c[0], 0.4 * c[1], c[2])

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

  sns.reset_defaults()

  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  color_index = 0

  # CDFs and PDFs
  pdf_cdf_container = report.add_html('<div style="display: flex; overflow-x: auto;"></div>')

  def add_pdf_cdf_sub_container(default):
    pdf_cdf_sub_container = report.add_html('<div class="pdf-cdf-sub-container"></div>', parent = pdf_cdf_container)
    report.add_html(f'''
      <p>
        Show
        <select class="pdf-cdf-selector">
          <option value="pdf" {'selected="true"' if (default == 'pdf') else ''}>probability density function (smoothed)</option>
          <option value="cdf" {'selected="true"' if (default == 'cdf') else ''}>cumulative density function</option>
        </select>
      </p>
    ''', parent = pdf_cdf_sub_container)
    return pdf_cdf_sub_container

  # Takeoff metrics
  pdf_cdf_sub_container = add_pdf_cdf_sub_container('pdf')
  for method in ['cdf', 'pdf']:
    plt.figure(figsize=(10,6))
    metric = 'billion_agis'
    plot = plot_ecdf if (method == 'cdf') else plot_epdf
    plot(results.scalar_metrics[metric], limits = [0, results.t_end - results.t_start], label = metric_id_to_human[metric], color = colors[color_index])
    plt.xlabel('Years')
    plt.ylabel(f'{method.upper()}\n(conditional on takeoff happening before {results.t_end})')
    plt.title(metric_id_to_human['billion_agis'])

    container = report.add_figure(parent = pdf_cdf_sub_container)
    report.add_class(container, method)
    if method == 'pdf':
      report.add_class(container, 'selected')
  color_index += 1

  # Timelines metrics
  bioanchors = BioAnchorsAGIDistribution()
  pdf_cdf_sub_container = add_pdf_cdf_sub_container('cdf')
  for method in ['cdf', 'pdf']:
    plt.figure(figsize=(10,6))
    for i, metric in enumerate(['rampup_start', 'agi_year']):
      plot = plot_ecdf if method == 'cdf' else plot_epdf
      plot(results.scalar_metrics[metric], limits = [results.t_start, results.t_end], label = metric_id_to_human[metric], color = colors[color_index + i])
    if method == 'cdf':
      plt.plot(bioanchors.years, bioanchors.cdf, color = '#e377c2', label = 'Bioanchors')
    plt.ylabel(method.upper())
    plt.xlabel('Year')
    plt.title('AI Timelines Metrics')
    plt.gca().legend(loc = (0.75, 0.15))

    container = report.add_figure(parent = pdf_cdf_sub_container)
    report.add_class(container, method)
    if method == 'cdf':
      report.add_class(container, 'selected')

  report.head.append(report.from_html('''
    <script>
      window.addEventListener('load', () => {
        for (let selector of document.querySelectorAll('.pdf-cdf-selector')) {
          let container = selector.parentElement.parentElement;
          selector.addEventListener('input', () => {
            for (let node of container.querySelectorAll('.selected')) {
              node.classList.remove('selected');
            }
            for (let node of container.querySelectorAll(`.${selector.value}`)) {
              node.classList.add('selected');
            }
          })
        }
      })
    </script>
  '''))

  report.head.append(report.from_html('''
    <style>
      .pdf-cdf-sub-container {
        min-width: 800px
      }

      .pdf-cdf-sub-container > .figure-container {
        display: none;
      }

      .pdf-cdf-sub-container > .figure-container.selected {
        display: unset;
      }
    </style>
  '''))

  # Plot trajectories
  metrics = {'biggest_training_run': 'Biggest training run'}
  for metric in metrics:
    results.state_metrics[metric] = np.stack([s[:len(results.timesteps)] for s in results.state_metrics[metric]])
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

  # "Years before AGI" tables

  header = report.add_header('"Years before AGI" tables', level = 3)
  header.append(report.generate_tooltip(f"""
    The distributions below are conditional on AGI actually happening before the end of the simulations ({results.t_end}).
    <br><br>
    When AGI happens early, we need to compute the value of the metrics before the start of the simulation
    (e.g., if AGI happens in {results.t_start + 5} years, we need their values for the year {results.t_start - 5} to generate the last column).
    We do so using a geometric extrapolation (except for the doubling time metrics, where we use a linear interpolation).
  """))

  metrics_before_agi_names = []

  id_to_name = get_variable_names()
  id_to_name['biggest_training_run'] = 'Biggest training run (measured in 2022-FLOP)'
  for i, (metric, table) in enumerate(results.metrics_before_agi_quantiles.items()):
    metric_name = metric.name
    if get_option('human_names'):
      metric_id = metric.name.split()[0]
      metric_name = f'{id_to_name[metric_id]} {metric.name[len(metric_id):].strip()}'
    metrics_before_agi_names.append(metric_name)

  metric_options = '\n'.join([f'<option value="{name}">{name}</option>' for name in metrics_before_agi_names])
  report.add_html(f'''
    <p>
      <select id="years-before-agi-selector">
        {metric_options}
      </select>
    </p>
  ''')

  years_before_agi_container = report.add_html('<div id="years-before-agi-container"></div>')

  for i, (metric, table) in enumerate(results.metrics_before_agi_quantiles.items()):
    dataframe = pd.DataFrame(table)
    table_container = report.add_data_frame(dataframe, show_index = False, float_format = metric.format_quantile, parent = years_before_agi_container)
    table_container.attrib['id'] = metrics_before_agi_names[i]
    if i == 0:
      report.add_class(table_container, 'selected')
    if metric.remarks:
      report.add_html(f'<p class="years-before-agi-remarks">{metric.remarks}</p>', parent = table_container)

  report.head.append(report.from_html('''
    <script>
      window.addEventListener('load', () => {
        let selector = document.getElementById('years-before-agi-selector');
        let container = document.getElementById('years-before-agi-container');
        selector.addEventListener('input', () => {
          let selected = container.querySelector('.selected');
          if (selected) {
            selected.classList.remove('selected');
          }
          document.getElementById(selector.value).classList.add('selected');
        })
      })
    </script>
  '''))

  report.head.append(report.from_html('''
    <style>
      #years-before-agi-container > * {
        display: none;
      }

      #years-before-agi-container > .selected {
        display: unset;
      }

      .years-before-agi-remarks {
        font-style: italic;
      }
    </style>
  '''))

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
    inputs_table = report.add_data_frame(
      results.parameter_table.drop(index = 'full_automation_requirements_training', columns = 'Type'),
      show_justifications = True,  nan_format = inputs_nan_format,
    )
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
    inputs_table = report.add_data_frame(results.parameter_table.drop(columns = 'Type'), show_justifications = True, nan_format = inputs_nan_format)

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

def filter_nans(x):
  return x[~np.isnan(x)]

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
