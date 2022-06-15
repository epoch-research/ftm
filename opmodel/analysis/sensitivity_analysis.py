"""
Sensitivity analysis.
"""

from . import log
from . import *

def sensitivity_analysis(report_file_path='sensitivity_analysis.html', report_dir_path=None):
  log.info('Retrieving parameters...')
  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")
  best_guess_parameters = {parameter : row["Best guess"] \
                           for parameter, row in parameter_table.iterrows()}

  parameter_count = len(parameter_table[parameter_table[['Conservative', 'Aggressive']].notna().all(1)])

  results = []
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

    results.append(result)

    current_parameter_index += 1

  log.info('Writing report...')

  results = pd.DataFrame(results)
  results = results.sort_values(by='delta', ascending=False)
  display(results)

  report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)
  report.add_data_frame(results)

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-o",
    "--output-file",
    default="sensitivity_analysis.html",
    help="Path of the output report (absolute or relative to the report directory)"
  )

  parser.add_argument(
    "-d",
    "--output-dir",
    default=Report.default_report_path(),
    help="Path of the output directory (will be create if it doesn't exist)"
  )

  args = parser.parse_args()
  sensitivity_analysis(report_file_path=args.output_file, report_dir_path=args.output_dir)

