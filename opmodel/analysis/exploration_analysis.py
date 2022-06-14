"""
Explore year by year.
"""

from . import *

def explore(exploration_target = 'compare'):
  log.info('Downloading parameters...')
  # Retrieve parameter table
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

  # Initialize simulations
  log.info('Initializing simulations...')
  low_model = SimulateTakeOff(**low_params)
  med_model = SimulateTakeOff(**med_params)
  high_model = SimulateTakeOff(**high_params)

  # Run simulations
  log.info('Running simulations...')
  low_model.run_simulation()
  med_model.run_simulation()
  high_model.run_simulation()

  # Print table of metrics
  log.info('Results...')
  low_results = {**{'type' : 'Conservative', 'value' : low_value}, **low_model.takeoff_metrics}
  med_results = {**{'type' : 'Best guess', 'value' : med_value}, **med_model.takeoff_metrics}
  high_results = {**{'type' : 'Aggressive', 'value' : high_value}, **high_model.takeoff_metrics}

  for metric in ['rampup_start', 'agi_year', 'doubling_times']:
    low_results[metric] = getattr(low_model, metric)
    med_results[metric] = getattr(med_model, metric)
    high_results[metric] = getattr(high_model, metric)

  results = [low_results, med_results, high_results]
  results = pd.DataFrame(results)
  display(results)

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
  plt.show()

  # Plot doubling times
  print("Model summaries")
  low_model.display_summary_table()
  med_model.display_summary_table()
  high_model.display_summary_table()

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
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-t", "--exploration_target", default="compare")
  args = parser.parse_args()

  explore(args.exploration_target)
