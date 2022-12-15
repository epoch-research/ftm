from . import log
from . import *

import matplotlib.ticker as mtick

def plot_compute_increase(scenario_group, title = "Compute increase over time", get_label = lambda scenario: scenario.name, show_legend = True):
  # Plot compute decomposition
  plt.figure(figsize=(14 if show_legend else 12, 8), dpi=80)

  for i, scenario in enumerate(scenario_group):
    plt.subplot(len(scenario_group), 1, i + 1)
    scenario.model.plot_compute_decomposition(new_figure=False)
    plt.ylabel(get_label(scenario))

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
    parameter_table = get_parameter_table()

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

  elif exploration_target == 'both_requirements_steepness':
    training_row = parameter_table.loc['training_requirements_steepness', :]
    runtime_row = parameter_table.loc['runtime_requirements_steepness', :]

    low_params = med_params.copy()
    low_params['runtime_requirements_steepness'] = runtime_row['Conservative']
    low_params['training_requirements_steepness'] = training_row['Conservative']

    high_params = med_params.copy()
    high_params['runtime_requirements_steepness'] = runtime_row['Aggressive']
    high_params['training_requirements_steepness'] = training_row['Aggressive']

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

  report.add_paragraph(f"<span style='font-weight:bold'>Exploration target:</span> {exploration_target}")

  # Print table of metrics
  low_results = {**{'type' : 'Conservative', 'value' : low_value},  **low_model.timeline_metrics, **low_model.takeoff_metrics}
  med_results = {**{'type' : 'Best guess', 'value' : med_value},  **low_model.timeline_metrics, **med_model.takeoff_metrics}
  high_results = {**{'type' : 'Aggressive', 'value' : high_value},  **low_model.timeline_metrics, **high_model.takeoff_metrics}

  for metric in ['doubling_times']:
    low_results[metric] = getattr(low_model, metric)
    med_results[metric] = getattr(med_model, metric)
    high_results[metric] = getattr(high_model, metric)

  results = [low_results, med_results, high_results]
  results = pd.DataFrame(results)
  report.add_data_frame(results)

  # Plot compute decomposition
  sub_ylabels = {
      'Conservative': 'Conservative',
      'Best guess':   'Best guess',
      'Aggressive':   'Aggressive',
  }
  ylabel = 'Scenario'

  if exploration_target not in ('compare', 'both_requirements_steepness'):
    param_names = get_param_names()
    ylabel = param_names[exploration_target]
    row = parameter_table.loc[exploration_target, :]
    for scenario in sub_ylabels:
      sub_ylabels[scenario] = format_float(row[scenario])

  plt.figure(figsize=(14, 8), dpi=80)

  plt.subplot(3, 1, 1)
  low_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel(sub_ylabels['Conservative'], fontweight='bold')
  plt.title(f"Compute increase over time")
  plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
  plt.gcf().tight_layout(rect=[0, 0.03, 1, 0.95])

  plt.subplot(3, 1, 2)
  med_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel(sub_ylabels['Best guess'], fontweight='bold')

  plt.subplot(3, 1, 3)
  high_model.plot_compute_decomposition(new_figure=False)
  plt.ylabel(sub_ylabels['Aggressive'], fontweight='bold')

  plt.gcf().text(-0.012, 0.5, ylabel, va='center', ha='center', rotation='vertical', fontsize=14, fontweight='bold')

  report.add_figure()

  # Plot requirements
  if exploration_target in ['both_requirements_steepness', 'training_requirements_steepness', 'runtime_requirements_steepness']:
    training = (exploration_target == 'training_requirements_steepness')

    requirements_low = low_model.automation_training_flops_goods if training else low_model.automation_runtime_flops_goods
    requirements_med = med_model.automation_training_flops_goods if training else med_model.automation_runtime_flops_goods
    requirements_high = high_model.automation_training_flops_goods if training else high_model.automation_runtime_flops_goods

    lims = [
      0.1 * min(
        np.min(requirements_low[1:]),
        np.min(requirements_med[1:]),
        np.min(requirements_high[1:]),
      ),
      10 * max(
        np.max(requirements_low[1:]),
        np.max(requirements_med[1:]),
        np.max(requirements_high[1:]),
      )
    ]

    def plot_requirements(reqs):
      reqs = reqs[1:]
      automatable = np.linspace(0, 100, len(reqs))

      reqs = np.append(np.insert(reqs, 0, lims[0]), lims[1])
      automatable = np.append(np.insert(automatable, 0, automatable[0]), automatable[-1])

      plt.plot(reqs, automatable)
      plt.xscale('log')
      plt.xlim(lims)
      plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.figure(figsize=(14, 8), dpi=80)
    plt.subplot(3, 1, 1)
    plot_requirements(requirements_low)
    plt.ylabel("Conservative")
    plt.title(f"{'Training' if training else 'Runtime'} requirements for goods and services")
    plt.tight_layout()
    plt.subplot(3, 1, 2)
    plot_requirements(requirements_med)
    plt.ylabel("Best guess")
    plt.subplot(3, 1, 3)
    plot_requirements(requirements_high)
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
  if med_params['runtime_training_tradeoff'] <= 0:
    del med_params['runtime_training_tradeoff']
    del high_params['runtime_training_tradeoff']
    del low_params['runtime_training_tradeoff']

    del med_params['runtime_training_max_tradeoff']
    del high_params['runtime_training_max_tradeoff']
    del low_params['runtime_training_max_tradeoff']

  report.add_header("Inputs", level = 3)
  input_parameters = pd.DataFrame(
    [low_params, med_params, high_params],
    index = ['Conservative', 'Best guess', 'Aggressive']
  ).transpose()
  report.add_data_frame(input_parameters, show_justifications = True)

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
  args = handle_cli_arguments(parser)
  explore(exploration_target=args.exploration_target, report_file_path=args.output_file, report_dir_path=args.output_dir)
