from . import log
from . import *

def add_scenario_group_to_report(scenario_group, report, exploration_target = 'compare'):
  results = []

  for scenario in scenario_group:
    row = {**{'type' : scenario_group.name, 'value' : scenario.name}, **scenario.model.takeoff_metrics}
    for metric in ['rampup_start', 'agi_year', 'doubling_times']:
      row[metric] = getattr(scenario.model, metric)
    results.append(row)

  results = pd.DataFrame(results)
  report.add_data_frame(results)

  plot_compute_increase()
  report.add_figure()

  # Plot doubling times
  report.add_header("Model summaries", level = 3)

  for scenario in scenario_group:
    report.add_header(scenario.name, level = 4)
    report.add_data_frame(scenario.model.get_summary_table())

  # Write down the parameters
  report.add_header("Inputs", level = 3)
  report.add_paragraph(f"<span style='font-weight:bold'>exploration_target:</span> {exploration_target}")
  input_parameters = pd.DataFrame(
    [scenario.params for scenario in scenario_group],
    index = [scenario.name for scenario in scenario_group]
  ).transpose()
  report.add_data_frame(input_parameters)

def plot_compute_increase(scenario_group, title = "Compute increase over time", show_legend = True):
  # Plot results
  metrics = ['gwp'] #, 'compute', 'hardware_performance', 'software', 'frac_gwp_compute', 'frac_training_compute']
  colors = ['red', 'orange', 'green']
  for metric in metrics:
    for i, scenario in enumerate(scenario_group):
      scenario.model.plot(metric, new_figure = (i > 0), line_color=colors[i % len(colors)])

  # Plot compute decomposition
  plt.figure(figsize=(14 if show_legend else 12, 8), dpi=80)

  for i, scenario in enumerate(scenario_group):
    plt.subplot(len(scenario_group), 1, i + 1)
    scenario.model.plot_compute_decomposition(new_figure=False)
    plt.ylabel(scenario.name)

    if i == 0:
      plt.title(title)
      if show_legend:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

  plt.tight_layout()

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
  plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
  plt.tight_layout();
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
