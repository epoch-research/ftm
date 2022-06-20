"""
Comparison of our model with Bio Anchors.
"""

from . import log
from . import *

def bioanchors_comparison(report_file_path='bioanchors_comparison.html', report_dir_path=None):
  log.info('Retrieving parameters...')

  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  log.info('Running our simulation...')
  model = SimulateTakeOff(**best_guess_parameters)
  model.run_simulation()

  log.info('Writing report...')
  report = Report(report_file_path = report_file_path, report_dir_path = report_dir_path)

  # Plot our forecast
  report.add_header("Our model", level = 3)
  plt.figure(figsize=(14, 8), dpi=80)
  model.plot_compute_decomposition_bioanchors_style(new_figure = False)
  report.add_figure()

  # Plot the bioanchors model forecast
  report.add_header("Bio Anchors model", level = 3)
  plt.figure(figsize=(14, 8), dpi=80)
  plot_bioanchors_model(t_step = model.t_step)
  report.add_figure()

  report.add_header("Simulation parameters", level = 3)

  report.add_data_frame(pd.DataFrame(best_guess_parameters, index = ['Best guess']).transpose())

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

def bioanchors_model(
    t_start                                = 2022,
    t_end                                  = 2100,
    t_step                                 = 0.1,  # Step duration, in years
    hardware_doubling_time                 = 2.5,  # In years
    software_doubling_time                 = 2.5,  # In years
    initial_training_investment            = 50,
    fast_training_investment_duration      = 23,   # In years
    fast_training_investment_doubling_time = 2.5,  # In years
    slow_training_investment_growth        = 1.03, # Per year
  ):

  hardware_growth = 2**(1/hardware_doubling_time)
  software_growth = 2**(1/software_doubling_time)
  fast_training_investment_growth = 2**(1/fast_training_investment_doubling_time)

  timesteps = np.arange(t_start, t_end, t_step)

  slow_training_investment_start_index = round(fast_training_investment_duration/t_step)

  fast_training_investment = \
      initial_training_investment * fast_training_investment_growth ** (timesteps[:slow_training_investment_start_index] - t_start)
  slow_training_investment = fast_training_investment[-1] * \
      slow_training_investment_growth ** (timesteps[slow_training_investment_start_index:] - timesteps[slow_training_investment_start_index-1])
  training_investment = np.concatenate([fast_training_investment, slow_training_investment])

  hardware = hardware_growth ** (timesteps - t_start)
  software = software_growth ** (timesteps - t_start)

  return timesteps, training_investment, hardware, software

def plot_bioanchors_model(*args, **kwargs):
  [timesteps, training_investment, hardware, software] = bioanchors_model(*args, **kwargs)

  plt.plot(timesteps, training_investment, label = 'Training compute investment', color = 'blue')
  plt.plot(timesteps, hardware, label = 'Hardware quality', color = 'orange')
  plt.plot(timesteps, software, label = 'Software', color = 'green')

  plt.yscale('log')
  utils.draw_oom_lines()


if __name__ == '__main__':
  parser = init_cli_arguments()
  args = parser.parse_args()
  bioanchors_comparison(report_file_path=args.output_file, report_dir_path=args.output_dir)
