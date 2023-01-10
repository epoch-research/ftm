# Execute from the root directory of the repo with  
#   python -m website.gen

# I should automate the whole process...

import re
import json
from ftm.core.utils import *
from ftm.core.model import *
from ftm.core.scenarios import *

important_params_and_metrics = pd.read_excel(
    get_input_workbook(),
    sheet_name = MOST_IMPORTANT_PARAMETERS_SHEET
)
important_params = important_params_and_metrics['Parameter id'].tolist()

def generate_sidebar_content():
  parameter_table = get_parameter_table(tradeoff_enabled=True)
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}
  param_names = get_param_names()

  test = 0
  important_parameters = []
  extra_parameters = []
  for param, value in best_guess_parameters.items():
    name = param_names[param]
    array = important_parameters if param in important_params else extra_parameters

    classes = 'input-parameter'
    additional_inputs = ''
    label_target = param

    if param in ['runtime_training_tradeoff', 'runtime_training_max_tradeoff']:
      enabled = best_guess_parameters['runtime_training_tradeoff'] > 0
      if not enabled:
        classes += ' disabled'
      id = "runtime_training_tradeoff_enabled_1" if param == 'runtime_training_tradeoff' else "runtime_training_tradeoff_enabled_2"
      additional_inputs += f' <input id="{id}" class="runtime_training_tradeoff_enabled" {"checked" if enabled else ""} type="checkbox">'
      label_target = id

    if param == 'runtime_training_tradeoff' and value < 0:
      value = 1

    array.append(f'''<div class="{classes}"><label for="{label_target}">{name}</label>{additional_inputs} <input class="input" id="{param}" value="{format_float(value)}"></div>''')
    test += 1

  # Basic params
  for p in important_parameters:
    print(p)

  # Extra params
  print('''
  <div class="handorgel additional-parameters">
    <h3 class="handorgel__header">
      <div class="handorgel__header__button" autofocus>
        Show additional parameters
        <span class="icon bi bi-plus-lg"></span>
      </div>
    </h3>
    <div class="handorgel__content">
''' + ('\n'.join(extra_parameters)) + '''
    </div>
  </div>
  ''')

def generate_dictionaries():
  print(f'let parameter_names = {json.dumps(get_param_names())};')
  print(f'let parameter_meanings = {json.dumps(get_parameters_meanings())};')
  print(f'let parameter_justifications = {json.dumps(get_parameter_justifications())};')
  print(f'let metric_names = {json.dumps(get_metric_names())};')
  print(f'let metric_meanings = {json.dumps(get_metrics_meanings())};')

  print(f'let variable_names = {json.dumps(get_variable_names())};')

  print()

  parameter_table = get_parameter_table(tradeoff_enabled=True)

  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  conservative_parameters = {
    parameter: row['Conservative'] if not np.isnan(row['Aggressive']) else row['Best guess'] for parameter, row in parameter_table.iterrows()
  }

  aggressive_parameters = {
    parameter: row['Aggressive'] if not np.isnan(row['Aggressive']) else row['Best guess'] for parameter, row in parameter_table.iterrows()
  }

  def handle_tradeoff(params):
    params['runtime_training_tradeoff_enabled'] = (params['runtime_training_tradeoff'] > 0)
    return params

  print(f'let conservative_parameters = {json.dumps(handle_tradeoff(conservative_parameters))};')
  print(f'let best_guess_parameters = {json.dumps(handle_tradeoff(best_guess_parameters))};')
  print(f'let aggressive_parameters = {json.dumps(handle_tradeoff(aggressive_parameters))};')

  log.level = Log.ERROR_LEVEL

  scenario_runner = ScenarioRunner()
  groups = scenario_runner.simulate_all_scenarios()

  parameters = {group.name: scenario.params for group in groups for scenario in group.scenarios if scenario.name == 'Best guess'}
  print()
  print(f'let short_timelines_best_guess = {json.dumps(handle_tradeoff(parameters["Very short timelines"]))};')
  print(f'let med_timelines_best_guess = {json.dumps(handle_tradeoff(parameters["Med timelines"]))};')
  print(f'let long_timelines_best_guess = {json.dumps(handle_tradeoff(parameters["Very long timelines"]))};')

def generate_arrays():
  print(f'let important_metrics = {json.dumps([x for x in important_params_and_metrics["Metric id"] if not pd.isnull(x)])};')

if __name__ == '__main__':
  # Handle CLI arguments
  parser = init_cli_arguments()
  args = handle_cli_arguments(parser)

  generate_sidebar_content()
  print()
  generate_dictionaries()
  print()
  generate_arrays()
