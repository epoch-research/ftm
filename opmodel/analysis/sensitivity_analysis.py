"""
Sensitivity analysis.
"""

from . import log
from . import *

class SensitivityAnalysisResults:
  def __init__(self, parameter_table = None, table = None):
    self.parameter_table = parameter_table
    self.table = table

def sensitivity_analysis():
  log.info('Retrieving parameters...')

  parameter_table = get_parameter_table()
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

    high_params = best_guess_parameters.copy()
    high_params[parameter] = row["Aggressive"]

    # Run simulations
    log.info(f"Running simulations for parameter '{parameter}' ({current_parameter_index + 1}/{parameter_count})...")

    log.info('  Conservative simulation...')
    low_model = SimulateTakeOff(**low_params)
    low_model.run_simulation()

    log.info('  Aggressive simulation...')
    high_model = SimulateTakeOff(**high_params)
    high_model.run_simulation()

    log.info('  Collecting results...')

    # Get results
    result = {
        'Parameter' : parameter,
        "Conservative value" : low_params[parameter],
        "Aggressive value" : high_params[parameter],
    }
    for takeoff_metric in low_model.takeoff_metrics:
      result[f"{takeoff_metric}"] = f"[{low_model.takeoff_metrics[takeoff_metric]:0.2f}, {high_model.takeoff_metrics[takeoff_metric]:0.2f}]"

    result["delta"] = np.abs(high_model.takeoff_metrics["combined"] - low_model.takeoff_metrics["combined"])
    
    # Add timelines metrics
    result["rampup_start"] = f"[{low_model.rampup_start:0.2f}, {high_model.rampup_start:0.2f}]"
    result["agi_date"] = f"[{low_model.agi_year:0.2f}, {high_model.agi_year:0.2f}]"

    # Add GWP doubling times
    result["GWP doubling times"] = f"[{low_model.doubling_times[:4]}, {high_model.doubling_times[:4]}]"
    result["doubling times"] = result["GWP doubling times"] # alias

    table.append(result)

    current_parameter_index += 1

  table = pd.DataFrame(table)
  table = table.set_index('Parameter').sort_values(by='delta', ascending=False)

  results = SensitivityAnalysisResults()
  results.parameter_table = parameter_table
  results.table = table

  return results

def write_sensitivity_analysis_report(report_file_path=None, report_dir_path=None, report=None):
  if report_file_path is None:
    report_file_path = 'sensitivity_analysis.html'

  #############################################################################
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO Remove this block and uncomment the line below it
  # TEST
  import os
  import pickle
  from . import CACHE_DIR
  cache_filename = os.path.join(CACHE_DIR, 'sensitivity_analysis.pickle')

  if not os.path.exists(cache_filename):
    results = sensitivity_analysis()
    with open(cache_filename, 'wb') as f:
      pickle.dump(results, f) # , pickle.HIGHEST_PROTOCOL)

  with open(cache_filename, 'rb') as f:
    results = pickle.load(f)
  # end testing
  #############################################################################

  # TODO uncomment this line
  #results = sensitivity_analysis()

  log.info('Writing report...')

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  report.add_data_frame(results.table)

  report.add_header("Inputs", level = 3)
  report.add_data_frame(results.parameter_table)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  parser = init_cli_arguments()
  args = parser.parse_args()
  write_sensitivity_analysis_report(report_file_path=args.output_file, report_dir_path=args.output_dir)

