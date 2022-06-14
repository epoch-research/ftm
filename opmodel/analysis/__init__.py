import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display

# In case we want to get rid of them
class Log:
  INFO_LEVEL = 1

  def __init__(self, level=0):
    self.level = level

  def info(self, msg):
    if self.level < Log.INFO_LEVEL: return
    print(msg)


log = Log(level=Log.INFO_LEVEL)
