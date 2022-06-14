"""
Monte carlo analysis.
"""

from . import *

from scipy.special import erfinv
from numpy.random import default_rng
rng = default_rng()
from matplotlib import cm

def credence_interval(low, high, alpha=0.9, kind='pos'):
  """ Returns a random number sampled from a lognormal
      with a given 70% confidence interval
      If frac, we transform the space to sample from odds space
  """

  if high < low:
    low, high = high, low
    
  if kind == "frac":
    low = low / (1. - low)
    high = high / (1. - high)

  elif kind == 'neg':
    low, high = -high, -low
  
  elif kind == "inv_frac":
    low, high = 1/high, 1/low
    low = low / (1. - low)
    high = high / (1. - high)
  
  elif not kind == "pos":
    raise ValueError(f"Unimplemented kind: {kind}")
    
  assert low < high
  
  inv_error = erfinv(alpha)
  mu = np.log(np.sqrt(low* high))
  sigma = (1./(np.sqrt(2)*inv_error))*np.log(np.sqrt(high/low))
  sample = rng.lognormal(mu, sigma)

  if kind == "frac":
    sample = 1. / (1. + 1. / sample)
  elif kind == "neg":
    sample = -sample
  elif kind == "inv_frac":
    sample = 1. + 1. / sample

  return sample 

# https://stackoverflow.com/questions/18313322/plotting-quantiles-median-and-spread-using-scipy-and-matplotlib
def plot_quantiles(ts, data, xlabel, ylabel, n_quantiles = 7, colormap = cm.Blues):
  # Fix overflows
  UPPER_BOUND = np.quantile(data, 0.95)
  data[data == 0.] = UPPER_BOUND
  data[data > UPPER_BOUND] = UPPER_BOUND

  # Compute quantiles
  n = len(ts)
  percentiles = np.linspace(0,100,n_quantiles)

  marks=np.zeros((n,n_quantiles))
  for i in range(n_quantiles):
    for t in range(n):
      marks[t,i]=np.percentile(data[:,t],percentiles[i])
  
  # Plot
  half = int((n_quantiles-1)/2)
  fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8,4))
  ax1.plot(ts, marks[:,half],color='k')
  for i in range(half):
    ax1.fill_between(ts, marks[:,i],marks[:,-(i+1)],color=colormap(i/half))

  ax1.set_title("Takeoff simulation", fontsize=15)
  ax1.set_yscale("log")
  ax1.tick_params(labelsize=11.5)
  ax1.set_xlabel(xlabel, fontsize=14)
  ax1.set_ylabel(ylabel, fontsize=14)
  fig.tight_layout()

n_trials = 100 #@param
scalar_metrics = ['rampup_start', 'agi_year']
state_metrics = ['gwp', 'biggest_training_run', 'compute']

scalar_metrics = {
    scalar_metric : [] for scalar_metric in scalar_metrics
}
for takeoff_metric in SimulateTakeOff.takeoff_metrics:
  scalar_metrics[takeoff_metric] = []

state_metrics = {
    state_metric : [] for state_metric in state_metrics
}

# Retrieve parameter table
parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
parameter_table = parameter_table.set_index("Parameter")

for trial in range(n_trials):
  # Sample parameters
  mc_params = {
      parameter : credence_interval(row['Conservative'], 
                                    row['Aggressive'], 
                                    kind=row['Type'])
                  if not np.isnan(row['Conservative'])\
                  and not np.isnan(row['Aggressive'])
                  else row['Best guess'] 
      for parameter, row in parameter_table.iterrows()
  }

  # Run simulation
  mc_model = SimulateTakeOff(**mc_params)
  mc_model.run_simulation()

  # Collect results  
  for scalar_metric in scalar_metrics:
    if scalar_metric in SimulateTakeOff.takeoff_metrics:
      metric_value = mc_model.takeoff_metrics[scalar_metric]
      assert metric_value >= 0, f"{scalar_metric} is negative!"
    else:
      metric_value = getattr(mc_model, scalar_metric)
    
    if scalar_metric == "rampup_start" and metric_value is None:
      metric_value = mc_model.t_end
    elif scalar_metric == "agi_year" and metric_value is None:
      metric_value = mc_model.t_end
    # if metric_value < 0.:
    #   print(f"{scalar_metric} is negative!")
    #   raise Exception
    scalar_metrics[scalar_metric].append(metric_value)

  for state_metric in state_metrics:
    metric_value = getattr(mc_model, state_metric)
    assert metric_value.shape == (mc_model.n_timesteps,)
    state_metrics[state_metric].append(metric_value)

# Plot trajectories
for state_metric in state_metrics:
  state_metrics[state_metric] = np.stack(state_metrics[state_metric])
  plot_quantiles(mc_model.timesteps, state_metrics[state_metric], "Year", state_metric)

# Display summary of scalar metrics
quantiles = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]
results = []
for q in quantiles:
  result = {"quantile" : q}
  for scalar_metric in scalar_metrics:
    result[scalar_metric] = np.quantile(scalar_metrics[scalar_metric],q)
  results.append(result)
results = pd.DataFrame(results)
display(results)
