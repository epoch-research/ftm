"""
Explore year by year.
"""

from . import *

exploration_target = 'compare' #@param {type:"string"}

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
low_model = SimulateTakeOff(**low_params)
med_model = SimulateTakeOff(**med_params)
high_model = SimulateTakeOff(**high_params)

# Run simulations
low_model.run_simulation()
med_model.run_simulation()
high_model.run_simulation()

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
