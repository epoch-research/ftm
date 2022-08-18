"""
Sensitivity analysis.
"""

from xml.etree import ElementTree as et

from . import log
from . import *

class SensitivityAnalysisResults:
  def __init__(self, parameter_table = None, table = None):
    self.parameter_table = parameter_table
    self.table = table

def sensitivity_analysis(quick_test_mode=False):
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

    # Add timelines metrics
    result["rampup_start"] = f"[{low_model.rampup_start:0.2f}, {med_model.rampup_start:0.2f}, {high_model.rampup_start:0.2f}]"
    result["agi_date"] = f"[{low_model.agi_year:0.2f}, {med_model.agi_year:0.2f}, {high_model.agi_year:0.2f}]"

    result["agi_date skew"] = skew(
      high_model.agi_year,
      med_model.agi_year,
      low_model.agi_year
    )

    # Add GWP doubling times
    result["GWP doubling times"] = f"[{low_model.doubling_times[:4]}, {med_model.doubling_times[:4]}, {high_model.doubling_times[:4]}]"
    result["doubling times"] = result["GWP doubling times"] # alias

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

def write_sensitivity_analysis_report(quick_test_mode=False, report_file_path=None, report_dir_path=None, report=None):
  if report_file_path is None:
    report_file_path = 'sensitivity_analysis.html'

  results = sensitivity_analysis(quick_test_mode)

  log.info('Writing report...')

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  table_container = report.add_data_frame(results.table)

  # Add "totals" table footer

  table = next(table_container.iter('table'))
  tfoot = et.fromstring('<tfoot><tr></tr></tfoot>')
  table.append(tfoot)

  columns = list(results.table.columns)
  skews = [c for c in columns if c.endswith(' skew')]

  tfoot.append(et.fromstring(f'<th>Totals</th>'))
  current_col = 0
  for skew in skews:
    skew_col = columns.index(skew)
    tfoot.append(et.fromstring(f'<th colspan="{skew_col - current_col}">Sum of {skew}s:</th>'))
    tfoot.append(et.fromstring(f'<th>{results.table[skew].sum():.2}</th>'))
    current_col = skew_col + 1
  tfoot.append(et.fromstring(f'<th colspan="{len(columns) - current_col}"></th>'))

  report.add_header("Inputs", level = 3)
  report.add_data_frame(results.parameter_table)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-q",
    "--quick-test-mode",
    action='store_true',
  )
  args = handle_cli_arguments(parser)
  write_sensitivity_analysis_report(report_file_path=args.output_file, report_dir_path=args.output_dir, quick_test_mode=args.quick_test_mode)

