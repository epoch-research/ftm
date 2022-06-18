"""
Explore year by year.
"""

from . import log
from . import *

def explore(exploration_target='compare', report_file_path=None, report_dir_path=None, parameter_table=None, report=None):
  if report_file_path is None:
    report_file_path = 'exploration_analysis.html'

  if parameter_table is None:
    # Retrieve parameter table
    log.info('Retrieving parameters...')
    parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
    parameter_table = parameter_table.set_index("Parameter")

  # Define parameters
  med_params = {
      parameter : row['Best guess']
      for parameter, row in parameter_table.iterrows()
  }

  if exploration_target == 'compare':

    low_params = {
        parameter : row['Conservative'] \
                    if not np.isnan(row['Conservative']) \
                    and row["Compare?"] == 'Y'
                    else row['Best guess']\
        for parameter, row in parameter_table.iterrows()
    }

    high_params = {
        parameter : row['Aggressive'] \
                    if not np.isnan(row['Aggressive']) \
                    and row["Compare?"] == 'Y'
                    else row['Best guess']\
        for parameter, row in parameter_table.iterrows()
    }

    low_value = 'Conservative'
    med_value = 'Best guess'
    high_value = 'Aggressive'

  else:
    row = parameter_table.loc[exploration_target, :]

    low_params = med_params.copy()
    low_params[exploration_target] = row['Conservative']

    high_params = med_params.copy()
    high_params[exploration_target] = row['Aggressive']

    low_value = row['Conservative']
    med_value = row['Best guess']
    high_value = row['Aggressive']

  # Run simulations
  log.info('Running simulations...')

  log.info('  Conservative simulation')
  low_model = SimulateTakeOff(**low_params)
  low_model.run_simulation()

  log.info('  Best guess simulation')
  med_model = SimulateTakeOff(**med_params)
  med_model.run_simulation()

  log.info('  Aggressive simulation')
  high_model = SimulateTakeOff(**high_params)
  high_model.run_simulation()

  log.info('Writing report...')
  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  # Print table of metrics
  low_results = {**{'type' : 'Conservative', 'value' : low_value}, **low_model.takeoff_metrics}
  med_results = {**{'type' : 'Best guess', 'value' : med_value}, **med_model.takeoff_metrics}
  high_results = {**{'type' : 'Aggressive', 'value' : high_value}, **high_model.takeoff_metrics}

  for metric in ['rampup_start', 'agi_year', 'doubling_times']:
    low_results[metric] = getattr(low_model, metric)
    med_results[metric] = getattr(med_model, metric)
    high_results[metric] = getattr(high_model, metric)

  results = [low_results, med_results, high_results]
  results = pd.DataFrame(results)
  report.add_data_frame(results)

  # Plot results
  metrics = ['gwp'] #, 'compute', 'hardware_performance', 'software', 'frac_gwp_compute', 'frac_training_compute']
  for metric in metrics:
    low_model.plot(metric, line_color='red')
    med_model.plot(metric, new_figure=False, line_color='orange')
    high_model.plot(metric, new_figure=False, line_color='green')

  # Plot compute decomposition
  plt.figure(figsize=(14, 8), dpi=80);
  plt.subplot(3, 1, 1)
  low_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel("Conservative")
  plt.title(f"Compute increase over time");
  plt.subplot(3, 1, 2)
  med_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel("Best guess")
  plt.subplot(3, 1, 3)
  high_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel("Aggressive")
  report.add_figure()

  # Plot doubling times
  report.add_header("Model summaries", level = 3)

  report.add_header("Conservative", level = 4)
  report.add_data_frame(low_model.get_summary_table())

  report.add_header("Best guess", level = 4)
  report.add_data_frame(med_model.get_summary_table())

  report.add_header("Aggressive", level = 4)
  report.add_data_frame(high_model.get_summary_table())

  # Write down the parameters
  report.add_header("Inputs", level = 3)
  report.add_paragraph(f"<span style='font-weight:bold'>exploration_target:</span> {exploration_target}")
  input_parameters = pd.DataFrame(
    [low_params, med_params, high_params],
    index = ['Conservative', 'Best guess', 'Aggressive']
  ).transpose()
  report.add_data_frame(input_parameters)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

def explore_year(year = 2020):
  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  # Run model
  model = SimulateTakeOff(**best_guess_parameters, t_step=1)
  model.run_simulation()

  # Plot things
  model.plot('gwp')
  model.plot_compute_decomposition()
  model.display_summary_table()
  model.display_takeoff_metrics()

  index = model.time_to_index(year)
  print(f"rampup = {model.rampup[index]}")
  print(f"hardware performance = {np.log10(model.hardware_performance[index])}")
  print(f"frac_gwp_compute = {model.frac_gwp_compute[index]}")
  print(f"hardware = {np.log10(model.hardware[index])}")
  print(f"software = {model.software[index]}")
  print(f"compute = {np.log10(model.compute[index])}")
  print(f"labour = {model.labour[index] / model.labour[0]}")
  print(f"capital = {model.capital[index] / model.capital[0] * 0.025}")
  print(f"automatable tasks goods = {model.automatable_tasks_goods[index]}")
  print(f"frac automatable tasks goods = {model.frac_automatable_tasks_goods[index]}")
  print(f"automatable tasks rnd = {model.automatable_tasks_rnd[index]}")
  print(f"frac automatable tasks rnd = {model.frac_automatable_tasks_rnd[index]}")
  print(f"gwp = {model.gwp[index]:e}")
  print(f"frac_capital_rnd = {model.frac_capital_hardware_rnd[index]}")
  print(f"frac_labour_rnd = {model.frac_labour_hardware_rnd[index]}")
  print(f"frac_compute_rnd = {model.frac_compute_hardware_rnd[index]}")
  print(f"rnd input hardware = {model.rnd_input_hardware[index] / model.rnd_input_hardware[0] * 0.003048307243707020}")
  print(f"cumulative input hardware = {model.cumulative_rnd_input_hardware[index] / model.rnd_input_hardware[0] * 0.003048307243707020}")
  print(f"ratio rnd input hardware = {model.rnd_input_hardware[0]**model.rnd_parallelization_penalty / model.cumulative_rnd_input_hardware[0]}")
  print(f"biggest_training_run = {np.log10(model.biggest_training_run[index])}")
  print(f"compute share goods = {model.compute_share_goods[index]}")

if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-t",
    "--exploration-target",
    default="compare",
    help="Choose 'compare' to compare aggressive, best guess and conservative scenario"
  )
  args = parser.parse_args()
  explore(exploration_target=args.exploration_target, report_file_path=args.output_file, report_dir_path=args.output_dir)
