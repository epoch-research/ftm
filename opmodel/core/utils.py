"""
Utilities.
"""

import os
import io
import re
import sys
import math
import yaml
import numpy as np
import pandas as pd
import urllib.request
from string import Template
import matplotlib.pyplot as plt
from openpyxl import load_workbook

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(MODULE_DIR, '../..'))
CONFIG_FILES = [
  os.path.join(PROJECT_DIR, 'config.yml'),
  os.path.join(PROJECT_DIR, 'local_config.yml'),
]

#--------------------------------------------------------------------------
# Poor person's mini configuration system
#--------------------------------------------------------------------------

config = None
config_loaded = False

def get_options():
  global config_loaded
  if not config_loaded:
    load_config_files()

  global config
  return config

def get_option(name):
  return get_options().get(name)

def set_option(name, value):
  global config_loaded
  if not config_loaded:
    load_config_files()

  global config
  config[name] = value

def load_config_files():
  configs = []
  for file in CONFIG_FILES:
    if os.path.exists(file):
      with open(file, "r") as f:
        result = yaml.load(f, Loader=yaml.FullLoader)
        if result:
          configs.append(result)

  global config
  config = merge_dicts(configs)
  expand_config_object(config)

  global config_loaded
  config_loaded = True

def merge_dicts(dicts):
  keys = []
  for d in dicts:
    for k in d.keys():
      if k not in keys:
        keys.append(k)

  merged = {}
  for k in keys:
    dicts_with_key = [d for d in dicts if k in d]

    for d in dicts_with_key:
      if type(d[k]) != type(dicts_with_key[0][k]):
        raise Exception(f'Conflict merging dictionaries (key {k})')

    if isinstance(dicts_with_key[0][k], dict):
      merged[k] = merge_dicts([d[k] for d in dicts_with_key])
    else:
      merged[k] = dicts_with_key[-1][k]

  return merged

def expand_config_object(config):
  context = {
    'project_dir': PROJECT_DIR,
  }
  _expand_config_object(config, context, '')

def _expand_config_object(obj, context, path):
  if isinstance(obj, (dict, list)):
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, value in items:
      if isinstance(value, (dict, list)):
        _expand_config_object(value, context, f'{path}{key}.')
        continue

      if isinstance(value, str):
        obj[key] = expand_string(value, context)
      context[path + key] = obj[key]

def expand_string(string, context):
  # Awfully hacky
  for key, value in context.items():
    string = string.replace(f'${{{key}}}', str(value))
  return string

#--------------------------------------------------------------------------
# Cached parameter retrieval
#--------------------------------------------------------------------------

cached_input_workbook = None
cached_ajeya_dist = None
cached_param_table = None
cached_rank_correlations = None
cached_timelines_parameters = None

def set_input_workbook(url):
  global input_workbook
  global cached_input_workbook
  global cached_param_table
  global cached_rank_correlations
  global cached_timelines_parameters

  # If it's a Google sheet url, we'll add the 'export?format=xlsx' ourselves
  pattern = r'https://docs.google.com/spreadsheets/d/([a-zA-Z0-9-_]*)/?.*'
  m = re.match(pattern, url)
  if m:
    workbook_id = m.group(1)
    url = f'https://docs.google.com/spreadsheets/d/{workbook_id}/export?format=xlsx'

  set_option('input_workbook', url)
  cached_input_workbook = None
  cached_param_table = None
  cached_rank_correlations = None
  cached_timelines_parameters = None

def get_input_workbook():
  global cached_input_workbook
  if cached_input_workbook is None:
    path = get_option('input_workbook')
    if re.match(r'^(http|https|file)://', path):
      gsheets_pattern = r'https://docs.google.com/spreadsheets/d/([a-zA-Z0-9-_]*)/?'
      m = re.match(gsheets_pattern, path)
      if m:
        workbook_id = m.group(1)
        path = f'https://docs.google.com/spreadsheets/d/{workbook_id}/export?format=xlsx'

      response = urllib.request.urlopen(path)
      cached_input_workbook = response.read()
    else:
      with open(path, 'rb') as f:
        cached_input_workbook = f.read()
  return cached_input_workbook

def get_parameter_table():
  global cached_param_table
  if cached_param_table is None:
    try:
      cached_param_table = pd.read_excel(get_input_workbook(), sheet_name = 'Parameters')
    except ValueError as e:
      print("Error reading the Excel file (you might want to check it's publicly accessible)", file=sys.stderr)
      raise e
    cached_param_table = cached_param_table.set_index("Parameter")
    cached_param_table.fillna(np.nan, inplace = True)
  return cached_param_table.copy()

def get_ajeya_dist():
  global cached_ajeya_dist
  if cached_ajeya_dist is None:
    # The sheet name is limited to 31 characters
    cached_ajeya_dist = pd.read_excel(get_input_workbook(), sheet_name = 'Ajeya distribution of automation FLOP'[:31])
  return cached_ajeya_dist.copy()

def get_rank_correlations():
  global cached_rank_correlations
  if cached_rank_correlations is None:
    cached_rank_correlations = pd.read_excel(get_input_workbook(), sheet_name = 'Rank correlations', skiprows = 2)
    cached_rank_correlations = cached_rank_correlations.set_index(cached_rank_correlations.columns[0])
  return cached_rank_correlations.copy()

def get_timelines_parameters():
  global cached_timelines_parameters
  if cached_timelines_parameters is None:
    cached_timelines_parameters = pd.read_excel(get_input_workbook(), sheet_name = 'Guess FLOP gap & timelines')
    cached_timelines_parameters = cached_timelines_parameters.set_index(cached_timelines_parameters.columns[0])
  return cached_timelines_parameters.copy()

def get_parameters_meanings():
  param_table = get_parameter_table()
  return param_table['Meaning'].to_dict()

def get_parameters_colors():
  colors = {}

  workbook = load_workbook(io.BytesIO(get_input_workbook()))
  sheet = workbook['Parameters']
  for row in sheet.iter_rows(min_row = 2, max_col = 1):
    cell = row[0]
    param = cell.value
    if not param: break
    colors[param] = f'#{cell.fill.bgColor.rgb[2:]}' # get rid of alpha

  return colors

def get_metrics_meanings():
  metric_meanings = pd.read_excel(get_input_workbook(), sheet_name = 'Output metrics')
  metric_meanings = metric_meanings.set_index(metric_meanings.columns[0])
  return metric_meanings['Meaning'].to_dict()

#--------------------------------------------------------------------------
# Misc
#--------------------------------------------------------------------------

def draw_oom_lines():
  low, high = plt.gca().get_ylim()
  for oom in range(math.floor(np.log10(low)), math.ceil(np.log10(high))):
    plt.axhline(10**oom, linestyle='dotted', color='black')

# Import display in non IPython environments
try:
  from IPython.display import display
except ModuleNotFoundError: 
  # As a fallback, just print
  def display(x):
    print(x)

class Log:
  ERROR_LEVEL = 1
  INFO_LEVEL  = 2

  def __init__(self, level=None):
    self.level = level if (level is not None) else INFO_LEVEL
    self.indentation_level = 0

  def indent(self):
    self.indentation_level += 1

  def deindent(self):
    self.indentation_level -= 1
    if self.indentation_level < 0: self.indentation_level = 0

  def info(self, *args, **kwargs):
    self.print(log.INFO_LEVEL, *args, **kwargs)

  def error(self, *args, **kwargs):
    self.print(log.ERROR_LEVEL, *args, **kwargs)

  def print(self, level, *args, **kwargs):
    if self.level < level: return

    string_buffer = io.StringIO()
    print(*args, **kwargs, file = string_buffer)
    string = string_buffer.getvalue()

    ends_in_newline = len(string) >= 1 and string[-1] == '\n'
    lines = string.splitlines()

    out = sys.stderr if (level == Log.ERROR_LEVEL) else sys.stdout
    for i, line in enumerate(lines):
      end = '' if (i == len(lines) - 1 and not ends_in_newline) else '\n'
      print((' ' * self.indentation_level) + line, file = out, end = end)

log = Log(level=Log.INFO_LEVEL)
