import re
import os
import sys
import inspect
import argparse
import pandas as pd
from opmodel.core.utils import *
from opmodel.core.opmodel import *
from openpyxl import load_workbook
from opmodel.report.report import Report

def compare(
    olde_sheet_path,
    parameter_table_url = 'https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/edit#gid=777689843',
    ):
  # -----------------------------------------------------------------------------
  # Modifify SimulateTakeOff
  # -----------------------------------------------------------------------------

  # Get source code
  code = inspect.getsource(SimulateTakeOff)

  # Modify the code (hackily)
  modified_code = []
  lines = iter(code.splitlines())
  for line in lines:
    # Change class name
    if re.match(r'^class SimulateTakeOff', line):
      line = re.sub(r'class SimulateTakeOff', r'class ModifiedSimulateTakeOff', line)

    # Get rid of asserts
    if re.match(r'^\s*assert', line):
      line = re.sub(r'(^\s*)', r'\1if False: ', line)

    # Allow infinite ceilings
    if re.match(r'^\s*def _update_rnd\(', line):
      # Find our target line
      while not re.match('^\s*ceiling_penalty =', line):
        modified_code.append(line)
        line = next(lines)
      line = re.sub(r'(\s*ceiling_penalty =)', r'\1 1 if performance_ceiling == np.inf else ', line)

    modified_code.append(line)

  modified_code = '\n'.join(modified_code)

  # Optionally save the modified code
  if True:
    with open(os.path.join(get_option('report_dir'), 'ModifiedSimulateTakeOff.py'), 'w') as f:
      f.write(modified_code)

  # Load the modified code
  exec(modified_code, globals(), globals())

  # -----------------------------------------------------------------------------
  # Load Ye Olde Sheet
  # -----------------------------------------------------------------------------
  olde_workbook = load_workbook(olde_sheet_path, data_only = True)
  olde_sheet = olde_workbook['CES production function']

  for c in reversed(olde_sheet['A']):
    if c.value is not None:
      last_year = c.value
      last_row = c.row
      break

  # -----------------------------------------------------------------------------
  # Simulation
  # -----------------------------------------------------------------------------

  # Retrieve parameter estimates from spreadsheet
  set_parameter_table_url(parameter_table_url)
  parameter_table = get_parameter_table()
  parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  # Convert strings to floats
  for p, v in parameters.items():
    if isinstance(v, str):
      if v.endswith('%'):
        parameters[p] = float(v[:-1])/100
      else:
        parameters[p] = float(v)

  parameters['n_labour_tasks'] = 40
  parameters['t_end'] = last_year + 1

  model = ModifiedSimulateTakeOff(**parameters, t_step=1)

  # Override the automation training FLOP for R&D
  model.automation_training_flops_rnd = 10**np.array([float(c.value) for row in olde_sheet['DI7:EV7'] for c in row])
  model.automation_training_flops_rnd = np.insert(model.automation_training_flops_rnd, 0, 1.0)

  model.run_simulation()

  # -----------------------------------------------------------------------------
  # Compare the results
  # -----------------------------------------------------------------------------
  def load_range(range):
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
    'agi_year'                   : load_range('BA54'),
    'full_rnd_automation_year'   : load_range('BB54'),
    'fast_growth_year'           : load_range('BC54'),
    'gwp'                        : load_range('P'),
    'compute'                    : 10**load_range('G'),
    'software'                   : load_range('AM'),
    'biggest_training_run'       : 10**load_range('BT'),
    'hardware_performance'       : 10**load_range(f'AD53:AD{last_row-1}'),
  }

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

  # Plain text output
  for label, var in year_variables.items():
    print(f'{label}')
    print(f'  {olde[var]}')
    print(f'  {modern[var]}')
    print()

  for label, var in state_variables.items():
    print(label)
    for i, year in enumerate(model.timesteps):
      print(f'  {year}: {olde[var][i]:.6e}')
      print(f'        {modern[var][i]:.6e}')
    print()

  # HTML output
  report = Report(report_file_path = 'excel_comparison.html')
  report.make_tables_scrollable = False

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
  print(f'Report stored in {report_path}')

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('local_olde_sheet_path')
  if len(sys.argv) < 2:
    print("You will have to download the Olde Sheet (https://docs.google.com/spreadsheets/d/1L38oMdU5cK2OipHTn3ZFiKhYgDXYKc1mfjGudsFrRUc/edit#gid=1889898348) as an Excel file and pass the local path of the downloaded file to this script. _Or_ make the sheet public and ask me to modify the script.\n", file = sys.stderr)
    parser.print_usage()
    sys.exit(1)
  args = parser.parse_args()

  olde_sheet_path = args.local_olde_sheet_path
  compare(olde_sheet_path)

