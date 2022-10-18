"""
Sensitivity analysis.
"""

from xml.etree import ElementTree as et

import os
import re
import gzip
import json
import dill as pickle
from multiprocessing import Pool

from . import log
from . import *
from ..stats.distributions import ParamsDistribution, PointDistribution, JointDistribution

method_human_names = {
  'one_at_a_time': 'One-at-a-time',
  'variance_reduction_on_margin': 'Variance reduction on the margin',
  'shapley_values': 'Variance reduction with Shapley values',
}

class SensitivityAnalysisResults:
  def __init__(self, parameter_table = None, table = None, analysis_params = None):
    self.parameter_table = parameter_table
    self.table = table
    self.analysis_params = analysis_params

def sensitivity_analysis(quick_test_mode = False, method = 'one_at_a_time'):
  if method == 'one_at_a_time':
    return one_at_a_time_comparison(quick_test_mode = quick_test_mode)

  if method == 'variance_reduction_on_margin':
    return variance_reduction_comparison(quick_test_mode = quick_test_mode, method = 'variance_reduction_on_margin')

  if method == 'shapley_values':
    return variance_reduction_comparison(quick_test_mode = quick_test_mode, method = 'shapley_values')

def variance_reduction_comparison(quick_test_mode = False, save_dir = None, restore_dir = None, method = 'variance_reduction_on_margin'):
  params_dist = ParamsDistribution(use_ajeya_dist = False, ignore_rank_correlations = True, resampling_method = 'resample_all')

  metric_names = SimulateTakeOff.takeoff_metrics + ['rampup_start', 'agi_year']
  parameters = [name for name, marginal in params_dist.marginals.items() if not isinstance(marginal, PointDistribution)]
  mean_samples = 100
  var_samples = 100
  shapley_samples = 1000
  processes = None

  if quick_test_mode:
    mean_samples = 50_000
    var_samples = 2
    processes = 4
    shapley_samples = 4
    parameters = [name for name, marginal in params_dist.marginals.items() if not isinstance(marginal, PointDistribution)]
    #parameters = ['flop_gap_training',]
    print(parameters)
    #parameters = ['full_automation_requirements_training', 'flop_gap_training',]
    #parameters = ['flop_gap_training',]

  log.info('Running importance analysis...')
  param_importances, param_stds = get_parameter_importance(
      params_dist, get_parameter_importance_metrics, parameters = parameters, metric_arguments = metric_names,
      mean_samples = mean_samples, var_samples = var_samples, shapley_samples = shapley_samples, processes = processes,
      save_dir = save_dir, restore_dir = restore_dir, method = method,
  )

  if save_dir:
    log.info(f'All the samples and metrics have been stored in {save_dir}')

  table = pd.DataFrame.from_dict(param_importances, orient = 'index', columns = metric_names)
  log.info()
  log.info('Importances:')
  log.info(table)

  std_table = pd.DataFrame.from_dict(param_stds, orient = 'index', columns = metric_names)
  log.info()
  log.info('Standard errors:')
  log.info(std_table)

  std_table = std_table.sort_values(by=main_metric, ascending=False, key = lambda x: table[main_metric])
  table = table.sort_values(by=main_metric, ascending=False)

  results = SensitivityAnalysisResults()
  results.parameter_table = params_dist.parameter_table[['Conservative', 'Best guess', 'Aggressive']]
  results.table = table
  results.std_table = std_table
  results.analysis_params = {
    'Number of inner (mean) samples': mean_samples,
    'Number of outer (variance) samples': var_samples,
  }

  if method == 'shapley_values':
    results.analysis_params['shapley_samples'] = shapley_samples

  return results

def get_parameter_importance_metrics(params, metric_names):
  model = SimulateTakeOff(**params, t_step = 1, dynamic_t_end = True)

  # Check that no goods task is automatable from the beginning (except for the first one)
  runtime_training_max_tradeoff = model.runtime_training_max_tradeoff if model.runtime_training_tradeoff is not None else 1.
  assert(not np.any(model.automation_training_flops_goods[1:] < model.initial_biggest_training_run * runtime_training_max_tradeoff))

  model.run_simulation()

  all_metrics = model.takeoff_metrics.copy()
  all_metrics["rampup_start"] = model.t_end if model.rampup_start is None else model.rampup_start
  all_metrics["agi_year"]     = model.t_end if model.agi_year is None else model.agi_year

  # Keep only the ones in metric_names (in that order)
  metrics = [all_metrics[name] for name in metric_names]

  return metrics

def one_at_a_time_comparison(quick_test_mode = False):
  log.info('Retrieving parameters...')

  parameter_table = get_parameter_table()
  parameter_table = parameter_table[['Conservative', 'Best guess', 'Aggressive']]
  best_guess_parameters = {parameter : row["Best guess"] \
                           for parameter, row in parameter_table.iterrows()}

  main_metric = 'combined'

  parameter_count = len(parameter_table[parameter_table[['Conservative', 'Aggressive']].notna().all(1)])

  table = []
  current_parameter_index = 0
  for parameter, row in parameter_table.iterrows():
    # Skip if there are no values for comparison
    if np.isnan(row["Conservative"]) \
    or np.isnan(row["Aggressive"]):
      continue

    # Define parameters
    low_params = best_guess_parameters.copy()
    low_params[parameter] = row["Conservative"]

    med_params = best_guess_parameters.copy()

    high_params = best_guess_parameters.copy()
    high_params[parameter] = row["Aggressive"]

    # Run simulations
    log.info(f"Running simulations for parameter '{parameter}' ({current_parameter_index + 1}/{parameter_count})...")

    log.info('  Conservative simulation...')
    low_model = SimulateTakeOff(**low_params, dynamic_t_end = True)
    low_model.run_simulation()

    log.info('  Best guess simulation...')
    med_model = SimulateTakeOff(**med_params, dynamic_t_end = True)
    med_model.run_simulation()

    log.info('  Aggressive simulation...')
    high_model = SimulateTakeOff(**high_params, dynamic_t_end = True)
    high_model.run_simulation()

    log.info('  Collecting results...')

    def skew(high, med, low):
      if high is None or med is None or low is None: return np.nan

      result = np.abs(med - low) - np.abs(high - med)
      if abs(result) < 1e-12: result = 0
      return result

    # Get results
    result = {
      'Parameter' : parameter,
    }
    for takeoff_metric in low_model.takeoff_metrics:
      result[f"{takeoff_metric}"] = f"[{low_model.takeoff_metrics[takeoff_metric]:0.2f}, {med_model.takeoff_metrics[takeoff_metric]:0.2f}, {high_model.takeoff_metrics[takeoff_metric]:0.2f}]"

      if takeoff_metric == 'billion_agis':
        result["billion_agis skew"] = skew(
          high_model.takeoff_metrics["billion_agis"],
          med_model.takeoff_metrics["billion_agis"],
          low_model.takeoff_metrics["billion_agis"]
        )

    result["importance"] = np.abs(high_model.takeoff_metrics[main_metric] - low_model.takeoff_metrics[main_metric])

    def format_year(model, year):
      if year is None or np.isnan(year): return f'> {model.t_end}'
      return f"{year:0.2f}"

    # Add timelines metrics
    result["rampup_start"] = f"[{format_year(low_model, low_model.rampup_start)}, {format_year(med_model, med_model.rampup_start)}, {format_year(high_model, high_model.rampup_start)}]"
    result["agi_year"] = f"[{format_year(low_model, low_model.agi_year)}, {format_year(med_model, med_model.agi_year)}, {format_year(high_model, high_model.agi_year)}]"

    result["agi_year skew"] = skew(
      high_model.agi_year,
      med_model.agi_year,
      low_model.agi_year
    )

    # Add GWP doubling times
    result["GWP doubling times"] = f"[{low_model.doubling_times[:4]}, {med_model.doubling_times[:4]}, {high_model.doubling_times[:4]}]"

    table.append(result)

    current_parameter_index += 1

    if quick_test_mode and current_parameter_index >= 1:
      break

  table = pd.DataFrame(table)
  table = table.set_index('Parameter').sort_values(by='importance', ascending=False)

  # Move the importance column to the beginning
  importance = table['importance']
  table.drop(labels = ['importance'], axis = 1, inplace = True)
  table.insert(0, 'importance', importance)

  results = SensitivityAnalysisResults()
  results.parameter_table = parameter_table
  results.table = table

  return results

def write_combined_sensitivity_analysis_report(
    methods=["one_at_a_time", "variance_reduction_on_margin"], quick_test_mode=False,
    report_file_path=None, report_dir_path=None, report=None,
    skip_inputs=False,
    analysis_results=[None,None], output_results_filenames=[None,None]
  ):

  if report_file_path is None:
    report_file_path = 'combined_sensitivity_analysis.html'

  log.info('Writing report...')

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  last_results = None

  for i in range(len(methods)):
    log.indent()
    log.info(f'Generating report for method {methods[i]}')

    saved_results = analysis_results[i] if (i < len(analysis_results)) else None
    filename_to_save_results = output_results_filenames[i] if (i < len(output_results_filenames)) else None
    last_results = write_sensitivity_analysis_report(
      report=report,
      method=methods[i], quick_test_mode=quick_test_mode,
      skip_inputs=True,
      analysis_results=saved_results, output_results_filename=filename_to_save_results
    )

    log.deindent()

  if not skip_inputs:
    report.add_header("Inputs", level = 3)
    inputs_table = report.add_data_frame(last_results.parameter_table, show_justifications = True, nan_format = inputs_nan_format)
    report.add_importance_selector(inputs_table, label = 'parameters', layout = 'vertical')

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

def write_sensitivity_analysis_report(
    method="one_at_a_time", quick_test_mode=False,
    report_file_path=None, report_dir_path=None, report=None,
    skip_inputs=False,
    analysis_results=None, output_results_filename=None
  ):

  if report_file_path is None:
    report_file_path = 'sensitivity_analysis.html'

  if isinstance(analysis_results, str):
    with open(analysis_results, 'rb') as f:
      analysis_results = pickle.load(f)

  results = analysis_results if analysis_results else sensitivity_analysis(quick_test_mode = quick_test_mode, method = method)

  if output_results_filename:
    with open(output_results_filename, 'wb') as f:
      pickle.dump(results, f)

  log.info('Writing report...')

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  report.add_header(method_human_names[method], level = 3)

  formatter = (lambda x: f'{max(x, 0):.0%}') if (method != 'one_at_a_time') else None
  table_container = report.add_data_frame(results.table, float_format = formatter)

  if method == 'one_at_a_time':
    def process_header(row, col, index_r, index_c, cell):
      if row == 0:
        if index_c in SimulateTakeOff.takeoff_metrics + ["rampup_start", "agi_year"]:
          cell.attrib['data-meaning-suffix'] = f'<br><br>Values of the metric as we set each parameter to their [conservative, best guess, aggressive] values.'
    report.apply_to_table(table_container, process_header)

  # Add "totals" table footer if there is some 'skew' column

  columns = list(results.table.columns)
  skews = [c for c in columns if c.endswith(' skew')]
  totals_row = None

  if skews:
    table = next(table_container.iter('table'))
    tfoot = et.fromstring('<tfoot><tr></tr></tfoot>')
    table.append(tfoot)

    tfoot.append(et.fromstring(f'<th>Totals</th>'))
    current_col = 0
    for skew in skews:
      total_skew = results.table[skew].sum(skipna = False)
      total_skew_str = f'{total_skew:0.2f}' if not np.isnan(total_skew) else 'NaN'

      skew_col = columns.index(skew)
      tfoot.append(et.fromstring(f'<th colspan="{skew_col - current_col}">Sum of {skew}s:</th>'))
      tfoot.append(et.fromstring(f'<th>{total_skew_str}</th>'))
      current_col = skew_col + 1
    tfoot.append(et.fromstring(f'<th colspan="{len(columns) - current_col}"></th>'))

    totals_row = len(table_container.findall('.//tr')) - 1

  most_important_metrics = get_most_important_metrics()
  most_important_parameters = get_most_important_parameters()

  def keep_cell(row, col, index_r, index_c, cell):
    fixed_cols = [0, 1, 2, 3] if method == 'one_at_a_time' else [0]
    col_condition = (col in fixed_cols) or (index_c in most_important_metrics or index_c in 'importance')
    row_condition = (row in [0]) or (index_r in most_important_parameters)
    return col_condition and row_condition

  report.add_importance_selector(table_container,
    label = 'parameters and metrics', layout = 'mixed',
    keep_cell = keep_cell,
  )

  if not skip_inputs:
    report.add_header("Inputs", level = 3)
    inputs_table = report.add_data_frame(results.parameter_table, show_justifications = True, nan_format = inputs_nan_format)
    report.add_importance_selector(inputs_table, label = 'parameters', layout = 'vertical')

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

  return results

def get_parameter_importance(
    dist, get_metrics,
    mean_samples = 4, var_samples = 10, shapley_samples = 1, parameters = None, metric_arguments = None, processes = None,
    save_dir = None, restore_dir = None, method = 'variance_reduction_on_margin'
  ):

  if parameters is None:
    parameters = [name for name, marginal in dist.marginals.items() if not isinstance(marginal, PointDistribution)]

  importances = {}
  stds = {}

  configured_g = lambda params: \
    g(dist, get_metrics, params, mean_samples = mean_samples, var_samples = var_samples,
      metric_arguments = metric_arguments, processes = processes, save_dir = save_dir, restore_dir = restore_dir)

  if method == 'variance_reduction_on_margin':
    log.info(f'Computing g_empty...')
    log.indent()
    g_empty, g_empty_std = configured_g([])
    log.deindent()
    for param in parameters:
      log.info(f'Computing importance of {param}...')
      log.indent()

      g_param, g_param_std = configured_g([param])

      importances[param] = 1 - g_param/g_empty
      stds[param] = (g_param/g_empty) * np.sqrt((g_param_std/g_param)**2 + (g_empty_std/g_empty)**2)

      log.deindent()

  elif method == 'shapley_values':
    log.info(f'Computing g_empty...')
    log.indent()
    g_empty, g_empty_std = configured_g([])
    log.deindent()

    K = shapley_samples
    for i, param in enumerate(parameters):
      log.info(f'Computing importance of {param}...')
      log.indent()
      r = 0
      v = 0
      for k in range(K):
        log.info(f'value {k:2}/{K}')
        permutation = np.random.permutation(parameters)
        param_index = permutation.tolist().index(param)

        # We are keeping the parameters sorted
        params        = [p for p in parameters if p in permutation[:param_index]]
        params_with_i = [p for p in parameters if p in permutation[:param_index+1]]

        log_level = log.level
        log.level = -1 # hackily deactivate logs

        g_params, g_params_std = configured_g(params)
        g_params_i, g_params_i_std = configured_g(params_with_i)

        r_k = (g_params - g_params_i)/g_empty
        v_k = (g_params_std**2 + g_params_i_std**2 + r_k**2 * g_empty_std**2)/g_empty**2

        r += r_k/K
        v += v_k/K**2

        log.level = log_level
      log.deindent()
      importances[param] = r
      stds[param] = np.sqrt(v)

  else:
    raise ValueError("method must be 'variance_reduction_on_margin' or 'shapley_values'")

  return importances, stds

def get_metric_values(get_metrics, dist, random_state, sample_count, conditions, metric_arguments):
  conditional_samples = dist.rvs(sample_count, random_state = random_state, conditions = conditions)
  metric_values = [get_metrics(s.to_dict(), metric_arguments) for _, s in conditional_samples.iterrows()]
  return metric_values, conditional_samples

def g(dist, get_metrics, parameters, mean_samples = 1, var_samples = 10, processes = None, metric_arguments = None, save_dir = None, restore_dir = None):
  """
  Compute E[var(get_metrics | parameters)]
  """

  if restore_dir:
    # Try to recover g from the cache files
    # This is wasteful, but it doesn't matter

    # Find the saved file we are interested in
    for file in os.listdir(restore_dir):
      m = re.match(r'g_([0-9]*)_0.gz', file)
      if m:
        index = int(m.group(1))
        with gzip.open(os.path.join(restore_dir, file), 'rb') as f:
          g_head = json.load(f)

        if g_head['fixed_params'] == parameters:
          # Found. We'll just read g from the files.
          log.info(f'Restoring g({parameters}) from {restore_dir}')
          vars = []
          for subfile in os.listdir(restore_dir):
            m = re.match(fr'g_{index}_([0-9]*).gz', subfile)
            if m:
              with gzip.open(os.path.join(restore_dir, subfile), 'rb') as f:
                g_i = json.load(f)
                vars += [np.var(iter['metric_values'], ddof = 1, axis = 0) for iter in g_i['iterations']]
          mean = np.mean(vars, axis = 0)
          std = np.std(vars, ddof = 1, axis = 0)/np.sqrt(len(vars))
          return mean, std
        

  if save_dir:
    if not os.path.exists(save_dir):
      os.makedirs(save_dir, exist_ok=True)

    save_file_index = 0
    for file in os.listdir(save_dir):
      m = re.match(r'g_([0-9]*)_?([0-9]*)*\.gz', file)
      if m and int(m.group(1)) >= save_file_index:
        save_file_index = int(m.group(1)) + 1

  if processes is None:
    processes = os.cpu_count()

  outer_samples = dist.rvs(mean_samples)

  seeder = np.random.SeedSequence()

  with Pool(processes = processes) as pool:
    workers = []

    max_batch_size = 1000
    save_file_subindex = 0
    metric_values_array = []
    conditional_samples_array = []

    def flush_batch():
      nonlocal conditional_samples_array
      nonlocal metric_values_array
      nonlocal save_file_subindex

      if len(metric_values_array) == 0:
        return

      save_dict = {}
      save_dict['params'] = list(dist.marginals.keys())
      save_dict['metric_arguments'] = metric_arguments
      save_dict['fixed_params'] = parameters
      save_dict['iterations'] = []
      for i in range(len(conditional_samples_array)):
        iteration = {}
        iteration['iter_sample'] = outer_samples.iloc[i].to_list()
        iteration['param_samples'] = conditional_samples_array[i].to_numpy().tolist()
        iteration['metric_values'] = np.array(metric_values_array[i]).tolist()
        save_dict['iterations'].append(iteration)

      filename = f'g_{save_file_index}_{save_file_subindex}.gz'
      with gzip.open(os.path.join(save_dir, filename), 'wb') as f:
        f.write(json.dumps(save_dict).encode('utf-8'))
      save_file_subindex += 1

      metric_values_array.clear()
      conditional_samples_array.clear()

    for row, outer_sample in outer_samples.iterrows():
      worker = pool.apply_async(get_metric_values, (
        get_metrics, dist, np.random.default_rng(seeder.spawn(1)[0]), var_samples, {name: outer_sample[name] for name in parameters}, metric_arguments
      ))
      workers.append(worker)

    vars = []

    for i, worker in enumerate(workers):
      log.info(f'Progress: {i:3}/{mean_samples}')
      metric_values, conditional_samples = worker.get()
      vars += [np.var(metric_values, ddof = 1, axis = 0)]

      if save_dir:
        conditional_samples_array.append(conditional_samples)
        metric_values_array.append(metric_values)
        if len(conditional_samples_array) >= max_batch_size:
          flush_batch()

    if save_dir:
      flush_batch()

    # Compute the mean and the standard error
    mean = np.mean(vars, axis = 0)
    std = np.std(vars, ddof = 1, axis = 0)/np.sqrt(len(vars))

    return mean, std

if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-q",
    "--quick-test-mode",
    action='store_true',
  )
  parser.add_argument(
    "-m",
    "--method",
    default = 'one_at_a_time',
    choices = [
      'one_at_a_time',
      'variance_reduction_on_margin',
      'shapley_values',
    ]
  )
  args = handle_cli_arguments(parser)
  write_sensitivity_analysis_report(report_file_path=args.output_file, report_dir_path=args.output_dir, quick_test_mode=args.quick_test_mode, method=args.method)

