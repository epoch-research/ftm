"""
Utilities.
"""

import io
import math
import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
from openpyxl import load_workbook

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
    if self.level < Log.INFO_LEVEL: return
    print(' ' * self.indentation_level, end = '')
    print(*args, **kwargs)

  def error(self, *args, **kwargs):
    if self.level < Log.ERROR_LEVEL: return
    print(' ' * self.indentation_level, end = '', file = sys.stderr)
    print(*args, **kwargs, file = sys.stderr)

log = Log(level=Log.INFO_LEVEL)

#--------------------------------------------------------------------------
# Cached parameter retrieval
#--------------------------------------------------------------------------

cached_omni_excel = None

def get_omni_excel():
  global cached_omni_excel
  if cached_omni_excel is None:
    response = urllib.request.urlopen('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=xlsx')
    cached_omni_excel = response.read()
  return cached_omni_excel

def get_parameter_table():
  parameter_table = pd.read_excel(get_omni_excel(), sheet_name = 'Parameters')
  parameter_table = parameter_table.set_index("Parameter")
  parameter_table.fillna(np.nan, inplace = True)
  return parameter_table

def get_timelines_parameters():
  timelines_parameters = pd.read_excel(get_omni_excel(), sheet_name = 'Guess FLOP gap & timelines')
  timelines_parameters = timelines_parameters.set_index(timelines_parameters.columns[0])
  return timelines_parameters

def get_parameters_meanings():
  param_table = get_parameter_table()
  return param_table['Meaning'].to_dict()

def get_parameters_colors():
  colors = {}

  workbook = load_workbook(io.BytesIO(get_omni_excel()))
  sheet = workbook['Parameters']
  for row in sheet.iter_rows(min_row = 2, max_col = 1):
    cell = row[0]
    param = cell.value
    if not param: break
    colors[param] = f'#{cell.fill.bgColor.rgb[2:]}' # get rid of alpha

  return colors

def get_metrics_meanings():
  metric_meanings = pd.read_excel(get_omni_excel(), sheet_name = 'Output metrics')
  metric_meanings = metric_meanings.set_index(metric_meanings.columns[0])
  return metric_meanings['Meaning'].to_dict()
