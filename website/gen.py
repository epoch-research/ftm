# Execute from the root directory of the repo with  
#   python -m website.gen

# I should automate the whole process...

import re
import json
from opmodel.core.utils import *
from opmodel.core.opmodel import *

important_params_and_metrics = pd.read_excel(
    get_input_workbook(),
    sheet_name = re.sub(r'[\[\]]', '', '[tom] Most important parameters and metrics')[:31]
)
important_params = important_params_and_metrics['Parameter id'].tolist()

def generate_sidebar_content():
  parameter_table = get_parameter_table()
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

    if param in ['runtime_training_tradeoff', 'runtime_training_max_tradeoff']:
      classes += ' disabled'
      additional_inputs += ' <input class="runtime_training_tradeoff_enabled" type="checkbox">'

    array.append(f'''<div class="{classes}"><label for="{param}">{name}</label>{additional_inputs} <input class="input" id="{param}" value="{value}"></div>''')
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

def generate_arrays():
  print(f'let important_metrics = {json.dumps(important_params_and_metrics["Metric id"].tolist())};')

if __name__ == '__main__':
  # Handle CLI arguments
  parser = init_cli_arguments()
  args = handle_cli_arguments(parser)

  generate_sidebar_content()
  print()
  generate_dictionaries()
  print()
  generate_arrays()
