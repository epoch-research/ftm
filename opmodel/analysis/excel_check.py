import re
import os
import inspect
from openpyxl import load_workbook

from . import log
from . import *

def write_excel_report(olde_sheet_url, report_file_path=None, report_dir_path=None):

  # -----------------------------------------------------------------------------
  # Modifify SimulateTakeOff
  # -----------------------------------------------------------------------------

  log.info(f'Modifying SimulateTakeOff (removing asserts)...')

  # Get source code
  code = inspect.getsource(SimulateTakeOff)

  # Get rid of asserts
  modified_code = []

  lines = iter(code.splitlines())
  for line in lines:
    # Change the class name
    if re.match(r'^class SimulateTakeOff', line):
      line = re.sub(r'class SimulateTakeOff', r'class ModifiedSimulateTakeOff', line)

    # Get rid of asserts
    if re.match(r'^\s*assert', line):
      line = re.sub(r'(^\s*)', r'\1if False: ', line)
    modified_code.append(line)

  modified_code = '\n'.join(modified_code)

  # Optionally save the modified code
  if True:
    with open(os.path.join(get_option('report_dir'), 'ModifiedSimulateTakeOff.py'), 'w', encoding = 'utf-8') as f:
      f.write(modified_code)

  # Load the modified code
  exec(modified_code, globals(), globals())

  # -----------------------------------------------------------------------------
  # Load Ye Olde Sheet
  # -----------------------------------------------------------------------------
  log.info(f'Retrieving the Olde Sheet...')
  olde_workbook = load_workbook(io.BytesIO(get_workbook(olde_sheet_url)), data_only = True)
  olde_sheet = olde_workbook['CES production function']

  t_step = olde_sheet['A8'].value
  n_labour_tasks = int(olde_sheet['JT8'].value)

  for c in reversed(olde_sheet['A']):
    if c.value is not None:
      last_year = c.value
      last_row = c.row
      break

  # -----------------------------------------------------------------------------
  # Simulation
  # -----------------------------------------------------------------------------

  # Retrieve parameter estimates
  log.info(f'Retrieving parameters...')
  parameter_table = get_parameter_table()
  parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  parameters['n_labour_tasks'] = n_labour_tasks
  parameters['t_end'] = last_year + 1

  model = ModifiedSimulateTakeOff(**parameters, t_step=t_step)

  # Override the automation training FLOP for R&D
  model.automation_training_flops_rnd = 10**np.array([float(c.value) for row in olde_sheet['DI7:EV7'] for c in row])
  model.automation_training_flops_rnd = np.insert(model.automation_training_flops_rnd, 0, 1.0)

  log.info(f'Running the simulation...')
  model.run_simulation()

  # -----------------------------------------------------------------------------
  # Compare the results and write report
  # -----------------------------------------------------------------------------
  def load_sheet_range(range):
    to_float = lambda x: np.nan if (x is None or x == '#REF!') else float(x)

    if re.match(r'^[A-Z]+$', range):
      col = range
      range = f'{col}54:{col}{last_row}'

    if ':' in range:
      # This is a real range
      array = []
      for row in olde_sheet[range]:
        for cell in row:
          array.append(to_float(cell.value))
      return np.array(array)
    else:
      # This is just a single cell
      return to_float(olde_sheet[range].value)

  olde = {
    'agi_year'                   : load_sheet_range('BA54'),
    'full_rnd_automation_year'   : load_sheet_range('BB54'),
    'fast_growth_year'           : load_sheet_range('BC54'),

    'timesteps'                  : load_sheet_range('A'),
    'gwp'                        : load_sheet_range('P'),
    'frac_gwp_compute'           : load_sheet_range('D'),
    'frac_compute_training'      : load_sheet_range('BR'),
    'compute'                    : 10**load_sheet_range('G'),
    'software'                   : load_sheet_range('AM'),
    'biggest_training_run'       : 10**load_sheet_range('BT'),
    'hardware_performance'       : 10**load_sheet_range(f'AD53:AD{last_row-1}'),
  }

  olde['compute_investment'] = olde['gwp'] * olde['frac_gwp_compute'] * t_step

  modern = {
    'agi_year'                  : float(model.index_to_time(np.argmax(model.frac_tasks_automated_goods))),
    'full_rnd_automation_year'  : float(model.index_to_time(np.argmax(model.frac_tasks_automated_rnd))),
    'fast_growth_year'          : float(model.index_to_time((np.argmax(np.log(model.gwp[1:]/model.gwp[:-1]) > 0.20)))),
    'gwp'                       : model.gwp,
    'compute'                   : model.compute,
    'software'                  : model.software,
    'biggest_training_run'      : model.biggest_training_run,
    'hardware_performance'      : model.hardware_performance,
  }

  year_variables = {
    'AGI': 'agi_year',
    'Full R&D automation': 'full_rnd_automation_year',
    'First 20% growth': 'fast_growth_year',
  }

  state_variables = {
    'GWP': 'gwp',
    'Compute': 'compute',
    'Biggest training run': 'biggest_training_run',
    'Hardware performance': 'hardware_performance',
  }

  # Write report
  log.info(f'Writing the report...')
  if report_file_path is None:
    report_file_path = 'excel_comparison.html'

  report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)
  report.make_tables_scrollable = False

  plt.figure(figsize=(14, 8), dpi=80)

  # Our model
  decomposition_data = model.plot_compute_decomposition(new_figure = False) # super hacky

  # Excel
  plot_compute_decomposition(
      decomposition_data['start_idx'], decomposition_data['end_idx'], decomposition_data['reference_idx'],
      olde['timesteps'], olde['compute_investment'], olde['hardware_performance'], olde['software'], olde['frac_compute_training']
  )

  # Hacky legend
  handles, labels = plt.gca().get_legend_handles_labels()

  vline_count = 0
  if model.rampup_start: vline_count += 1
  if model.rampup_mid:   vline_count += 1
  if model.agi_year:     vline_count += 1

  new_handles = handles[:4]
  new_labels = labels[:4]
  vline_handles = handles[4:4+vline_count]
  vline_labels = labels[4:4+vline_count]
  old_handles = handles[4+vline_count:]
  old_labels = labels[4+vline_count:]

  old_legend = plt.legend(old_handles, old_labels, bbox_to_anchor = (1.02, 1), borderaxespad = 0, title = 'Olde model')
  new_legend = plt.legend(new_handles, new_labels, bbox_to_anchor = (1.02, 0.80), borderaxespad = 0, title = 'New model')
  vline_legend = plt.legend(vline_handles, vline_labels, bbox_to_anchor = (1.02, 0.60), borderaxespad = 0)

  old_legend.set_frame_on(False)
  new_legend.set_frame_on(False)
  vline_legend.set_frame_on(False)

  old_legend._legend_box.align = "left"
  new_legend._legend_box.align = "left"
  vline_legend._legend_box.align = "left"

  plt.gcf().add_artist(old_legend)
  plt.gcf().add_artist(new_legend)
  plt.gcf().add_artist(vline_legend)

  report.add_figure()

  year_variables_table = {label: [olde[var], modern[var]] for label, var in year_variables.items()}
  year_variables_table = pd.DataFrame(year_variables_table, index = ['Excel', 'Python'])
  report.add_data_frame(year_variables_table)

  for label, var in state_variables.items():
    container = report.add_html('<div class="header-container"></div>')
    header = report.add_header(label, level = 4, parent = container)
    table = {year: [olde[var][i], modern[var][i]] for i, year in enumerate(model.timesteps)}
    table = pd.DataFrame(table, index = ['Excel', 'Python'])
    report.add_data_frame(table)

  report.add_html('''
    <style>
      .header-container {
        width: 100%;
      }

      h4 {
        position: sticky;
        left: 1em;
        display: inline-block;
        margin-bottom: 2px;
      }
    </style>
  ''', parent = report.head)

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

  return model

def plot_compute_decomposition(start_idx, end_idx, reference_idx, timesteps, compute_investment, hardware_performance, software, frac_compute_training):
  linestyle = '--'

  clip = lambda x: x[start_idx:end_idx]
  clip_and_normalize = lambda x: clip(x)/x[reference_idx]

  plt.plot(clip(timesteps), clip_and_normalize(compute_investment), label = '$ on FLOP globally', color = 'blue', linestyle = linestyle)
  plt.plot(clip(timesteps), clip_and_normalize(hardware_performance), label = 'Hardware quality', color = 'orange', linestyle = linestyle)
  plt.plot(clip(timesteps), clip_and_normalize(software), label = 'Software', color = 'green', linestyle = linestyle)
  plt.plot(clip(timesteps), clip_and_normalize(frac_compute_training), label = 'Fraction global FLOP on training', color = 'red', linestyle = linestyle)
  plt.yscale('log')

  draw_oom_lines()


if __name__ == '__main__':
  parser = init_cli_arguments()

  parser.add_argument(
    "-s",
    "--olde-sheet",
    type=str,
    default='https://docs.google.com/spreadsheets/d/1rMMkUsuxzuRqa9R5Foqu7XSinvU6-zow2FzX96PmSrI/edit?usp=sharing',
  )

  args = handle_cli_arguments(parser)

  write_excel_report(
    report_file_path=args.output_file,
    report_dir_path=args.output_dir,
    olde_sheet_url=args.olde_sheet,
  )
