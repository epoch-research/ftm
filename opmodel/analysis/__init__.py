import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display
from ..report.report import Report

class Log:
  INFO_LEVEL = 1

  def __init__(self, level=0):
    self.level = level

  def info(self, *args, **kwargs):
    if self.level < Log.INFO_LEVEL: return
    print(*args, **kwargs)

log = Log(level=Log.INFO_LEVEL)
