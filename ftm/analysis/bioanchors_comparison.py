"""
Comparison of our model with Bio Anchors.
"""

from . import log
from . import *

from matplotlib.patches import Rectangle

def bioanchors_comparison(report_file_path=None, report_dir_path=None):
  if report_file_path is None:
    report_file_path = 'bioanchors_comparison.html'

  log.info('Retrieving parameters...')

  t_end = 2050

  parameter_table = get_parameter_table()
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  log.info('Running our simulation...')
  model = SimulateTakeOff(**best_guess_parameters, t_end = t_end)
  model.run_simulation()

  log.info('Writing report...')
  report = Report(report_file_path = report_file_path, report_dir_path = report_dir_path)

  # Plot our forecast
  report.add_header("Our model vs bioanchors", level = 3)
  plt.figure(figsize=(14, 8), dpi=80)

  model.plot_compute_decomposition_bioanchors_style(new_figure = False)
  our_ylims = plt.gca().get_ylim()

  # Plot the bioanchors model forecast
  plot_bioanchors_model(t_step = model.t_step, t_end = model.t_end, ylims = our_ylims)

  # Hacky legend
  handles, labels = plt.gca().get_legend_handles_labels()

  our_handles = handles[:3]
  our_labels = labels[:3]
  bio_handles = handles[3:]
  bio_labels = labels[3:]

  our_legend = plt.legend(our_handles, our_labels, bbox_to_anchor = (1.02, 1), borderaxespad = 0, title = 'Our model')
  bio_legend = plt.legend(bio_handles, bio_labels, bbox_to_anchor = (1.02, 0.85), borderaxespad = 0, title = 'Bioanchors')

  our_legend.set_frame_on(False)
  bio_legend.set_frame_on(False)

  our_legend._legend_box.align = "left"
  bio_legend._legend_box.align = "left"

  plt.gcf().add_artist(our_legend)
  plt.gcf().add_artist(bio_legend)


  plt.xticks(np.arange(2020, 2055, 5))

  report.add_figure()

  # Wrap up
  report.add_header("Simulation parameters", level = 3)

  report.add_data_frame(pd.DataFrame(best_guess_parameters, index = ['Best guess']).transpose())

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

def bioanchors_model(
    t_start                                = 2022,
    t_end                                  = 2100,
    t_step                                 = 0.1,   # Step duration, in years

    initial_hardware                       = 1.7/3, # Compared to our model
    hardware_doubling_time                 = 2.5,   # In years

    software_doubling_start                = 2025,  # When the software starts doubling
    software_doubling_time                 = 2.5,   # In years
    software_ceiling                       = 1e3,   # Max software performance from 2022 levels

    initial_training_investment            = 50,    # Compared to our model
    fast_training_investment_duration      = 23,    # In years
    fast_training_investment_doubling_time = 2.5,   # In years
    slow_training_investment_growth        = 1.03,  # Per year
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

  hardware = initial_hardware * hardware_growth ** (timesteps - t_start)

  software_timesteps = np.arange(software_doubling_start, t_end, t_step)
  software = software_growth ** (software_timesteps - software_doubling_start)
  software = np.clip(software, a_min = None, a_max = software_ceiling)

  return timesteps, training_investment, hardware, software_timesteps, software

def plot_bioanchors_model(*args, ylims = None, **kwargs):
  [timesteps, training_investment, hardware, software_timesteps, software] = bioanchors_model(*args, **kwargs)

  linestyle = '--'

  plt.plot(timesteps, training_investment, label = 'Training compute investment', color = 'purple', linestyle = linestyle)
  plt.plot(timesteps, hardware, label = 'Hardware quality', color = 'orange', linestyle = linestyle)
  plt.plot(software_timesteps, software, label = 'Software', color = 'green', linestyle = linestyle)
  plt.yscale('log')

  if ylims is not None:
    plt.gca().set_ylim(ylims)

  draw_oom_lines()


if __name__ == '__main__':
  parser = init_cli_arguments()
  args = handle_cli_arguments(parser)
  bioanchors_comparison(report_file_path=args.output_file, report_dir_path=args.output_dir)
