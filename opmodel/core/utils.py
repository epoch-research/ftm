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

