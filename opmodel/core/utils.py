"""
Utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import math

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

