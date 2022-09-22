"""
Sensitivity analysis.
"""

from xml.etree import ElementTree as et

import os
import re
import gzip
import json
from multiprocessing import Pool

from . import log
from . import *
from ..stats.distributions import ParamsDistribution, PointDistribution, JointDistribution

class SensitivityAnalysisResults:
  def __init__(self, parameter_table = None, table = None, analysis_params = None):
    self.parameter_table = parameter_table
    self.table = table
    self.analysis_params = analysis_params

def sensitivity_analysis(quick_test_mode = False, method = 'point_comparison'):
  if method == 'point_comparison':
    return point_comparison(quick_test_mode = quick_test_mode)

  if method == 'variance_reduction_on_margin':
    return variance_reduction_comparison(quick_test_mode = quick_test_mode, method = 'variance_reduction_on_margin')

  if method == 'shapley_values':
    return variance_reduction_comparison(quick_test_mode = quick_test_mode, method = 'shapley_values')

def variance_reduction_comparison(quick_test_mode = False, save_dir = None, method = 'variance_reduction_on_margin'):
  params_dist = ParamsDistribution()

  metric_names = SimulateTakeOff.takeoff_metrics + ['rampup_start', 'agi_year']
  parameters = [name for name, marginal in params_dist.marginals.items() if not isinstance(marginal, PointDistribution)]
  mean_samples = 100
  var_samples = 100
  shapley_samples = 1000

  if quick_test_mode:
    mean_samples = 10
    shapley_samples = 4
    var_samples = 10
    parameters = parameters[:5]

  log.info('Running simulations...')
  param_importance = get_parameter_importance(
      params_dist, parameter_importance_metrics, parameters = parameters, metric_arguments = metric_names,
      mean_samples = mean_samples, var_samples = var_samples, shapley_samples = shapley_samples,
      save_dir = save_dir, method = method,
  )

  if save_dir:
    log.info(f'All the samples and metrics have been stored in {save_dir}')

  table = pd.DataFrame.from_dict(param_importance, orient = 'index', columns = metric_names)
  log.info(table)

  results = SensitivityAnalysisResults()
  results.parameter_table = params_dist.parameter_table
  results.table = table
  results.analysis_params = {
    'mean_samples': mean_samples,
    'var_samples': var_samples,
  }

  if method == 'shapley_values':
    results.analysis_params['shapley_samples'] = shapley_samples

  return results

def parameter_importance_metrics(params, metric_names):
  model = SimulateTakeOff(**params, t_step = 1)

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

def point_comparison(quick_test_mode = False):
  log.info('Retrieving parameters...')

  parameter_table = get_parameter_table()
  parameter_table = parameter_table[['Conservative', 'Best guess', 'Aggressive']]
  best_guess_parameters = {parameter : row["Best guess"] \
                           for parameter, row in parameter_table.iterrows()}

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
    low_model = SimulateTakeOff(**low_params)
    low_model.run_simulation()

    log.info('  Best guess simulation...')
    med_model = SimulateTakeOff(**med_params)
    med_model.run_simulation()

    log.info('  Aggressive simulation...')
    high_model = SimulateTakeOff(**high_params)
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
        "Conservative value" : low_params[parameter],
        "Best guess value" : med_params[parameter],
        "Aggressive value" : high_params[parameter],
    }
    for takeoff_metric in low_model.takeoff_metrics:
      result[f"{takeoff_metric}"] = f"[{low_model.takeoff_metrics[takeoff_metric]:0.2f}, {med_model.takeoff_metrics[takeoff_metric]:0.2f}, {high_model.takeoff_metrics[takeoff_metric]:0.2f}]"

      if takeoff_metric == 'billion_agis':
        result["billion_agis skew"] = skew(
          high_model.takeoff_metrics["billion_agis"],
          med_model.takeoff_metrics["billion_agis"],
          low_model.takeoff_metrics["billion_agis"]
        )


    result["delta"] = np.abs(high_model.takeoff_metrics["combined"] - low_model.takeoff_metrics["combined"])

    def format_year(model, year):
      if year is None or np.isnan(year): return f'> {model.t_end}'
      return f"{year:0.2f}"

    # Add timelines metrics
    result["rampup_start"] = f"[{format_year(low_model, low_model.rampup_start)}, {format_year(med_model, med_model.rampup_start)}, {format_year(high_model, high_model.rampup_start)}]"
    result["agi_date"] = f"[{format_year(low_model, low_model.agi_year)}, {format_year(med_model, med_model.agi_year)}, {format_year(high_model, high_model.agi_year)}]"

    result["agi_date skew"] = skew(
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
  table = table.set_index('Parameter').sort_values(by='delta', ascending=False)

  results = SensitivityAnalysisResults()
  results.parameter_table = parameter_table
  results.table = table

  return results

def write_sensitivity_analysis_report(method="point_comparison", quick_test_mode=False, report_file_path=None, report_dir_path=None, report=None):
  if report_file_path is None:
    report_file_path = 'sensitivity_analysis.html'

  results = sensitivity_analysis(quick_test_mode = quick_test_mode, method = method)

  log.info('Writing report...')

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  header_lines = []
  header_lines.append('<p>')
  header_lines.append(f'Method: {method}')
  if results.analysis_params:
    for name, value in results.analysis_params.items():
      header_lines.append(f'<br/>{name}: {value}')
  header_lines.append('</p>')
  report.add_html('\n'.join(header_lines))

  table_container = report.add_data_frame(results.table)

  # Add "totals" table footer if there are some 'skew' column

  columns = list(results.table.columns)
  skews = [c for c in columns if c.endswith(' skew')]

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

  report.add_header("Inputs", level = 3)
  report.add_data_frame(results.parameter_table, show_justifications = True)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

def get_parameter_importance(
    dist, metric,
    mean_samples = 4, var_samples = 10, shapley_samples = 1, parameters = None, metric_arguments = None,
    save_dir = None, method = 'variance_reduction_on_margin'
  ):

  if parameters is None:
    parameters = [name for name, marginal in dist.marginals.items() if not isinstance(marginal, PointDistribution)]

  importance = {}

  configured_g = lambda params: \
    g(dist, metric, params, mean_samples = mean_samples, var_samples = var_samples, metric_arguments = metric_arguments, save_dir = save_dir)

  if method == 'variance_reduction_on_margin':
    log.info(f'Computing g_empty...')
    log.indent()
    g_empty = configured_g([])
    log.deindent()
    for param in parameters:
      log.info(f'Running simulations for {param}...')
      log.indent()
      importance[param] = 1 - configured_g([param])/g_empty
      log.deindent()

  elif method == 'shapley_values':
    log.info(f'Computing g_empty...')
    log.indent()
    g_empty = configured_g([])
    log.deindent()

    K = shapley_samples
    for i, param in enumerate(parameters):
      log.info(f'Running simulations for {param}...')
      log.indent()
      r = 0
      for k in range(K):
        log.info(f'value {k:2}/{K}')
        permutation = np.random.permutation(parameters)
        param_index = permutation.tolist().index(param)

        # We are keeping the parameters sorted
        params        = [p for p in parameters if p in permutation[:param_index]]
        params_with_i = [p for p in parameters if p in permutation[:param_index+1]]

        log_level = log.level
        log.level = -1
        r_k = (configured_g(params) - configured_g(params_with_i))/g_empty
        r += r_k/K
        log.level = log_level
      log.deindent()
      importance[param] = r

  else:
    raise ValueError("method must be one of 'variance_reduction_on_margin' or 'shapley_values'")

  return importance

def get_metric_values(metric, conditional_samples, metric_arguments):
  metric_values = [metric(s.to_dict(), metric_arguments) for _, s in conditional_samples.iterrows()]
  return metric_values

def g(dist, metric, parameters, mean_samples = 1, var_samples = 10, processes = None, metric_arguments = None, save_dir = None):
  """
  Compute E[var(metric | parameters)]
  """
  if processes is None:
    processes = os.cpu_count()

  outer_samples = dist.rvs(mean_samples)

  with Pool(processes = processes) as pool:
    workers = []

    conditional_samples_array = []
    for row, outer_sample in outer_samples.iterrows():
      conditional_samples = dist.rvs(var_samples, conditions = {name: outer_sample[name] for name in parameters})
      conditional_samples_array.append(conditional_samples)
      worker = pool.apply_async(get_metric_values, (metric, conditional_samples, metric_arguments))
      workers.append(worker)

    metric_values_array = []
    for i, worker in enumerate(workers):
      log.info(f'Progress: {i:3}/{mean_samples}')
      metric_values = worker.get()
      metric_values_array.append(metric_values)

    if save_dir:
      # Optionally store this information

      save_dict = {}
      save_dict['params'] = list(dist.marginals.keys())
      save_dict['metric_arguments'] = metric_arguments
      save_dict['fixed_params'] = parameters
      save_dict['iterations'] = []
      for i in range(mean_samples):
        iteration = {}
        iteration['iter_sample'] = outer_samples.iloc[i].to_list()
        iteration['param_samples'] = conditional_samples_array[i].to_numpy().tolist()
        iteration['metric_values'] = np.array(metric_values_array[i]).tolist()
        save_dict['iterations'].append(iteration)

      if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

      last_file_number = -1
      for file in os.listdir(save_dir):
        m = re.match(r'g_([0-9]*).gz', file)
        if m and int(m.group(1)) > last_file_number:
          last_file_number = int(m.group(1))

      filename = f'g_{last_file_number+1}.gz'
      with gzip.open(os.path.join(save_dir, filename), 'wb') as f:
        f.write(json.dumps(save_dict).encode('utf-8'))

    # Compute the mean

    means = [np.var(metric_values, ddof = 1, axis = 0) for metric_values in metric_values_array]
    mean = np.mean(means, axis = 0)

    return mean

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
    default = 'point_comparison',
    choices = [
      'point_comparison',
      'variance_reduction_on_margin',
      'shapley_values',
    ]
  )
  args = handle_cli_arguments(parser)
  write_sensitivity_analysis_report(report_file_path=args.output_file, report_dir_path=args.output_dir, quick_test_mode=args.quick_test_mode, method=args.method)

