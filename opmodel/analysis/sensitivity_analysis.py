"""
Sensitivity analysis.
"""

from . import *

parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
parameter_table = parameter_table.set_index("Parameter")
best_guess_parameters = {parameter : row["Best guess"] \
                         for parameter, row in parameter_table.iterrows()}
results = []
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
  low_model = SimulateTakeOff(**low_params)
  high_model = SimulateTakeOff(**high_params)

  low_model.run_simulation()
  high_model.run_simulation()

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
    
results = pd.DataFrame(results)
results = results.sort_values(by='delta', ascending=False)
display(results)
