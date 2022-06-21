import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display
from ..core import utils
from ..report.report import Report

# Some constants
CACHE_DIR = '_cache_/diffy'
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(MODULE_DIR, '..', '..', CACHE_DIR))

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

def init_cli_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-o",
    "--output-file",
    default=None,
    help="Path of the output report (absolute or relative to the report directory)"
  )

  parser.add_argument(
    "-d",
    "--output-dir",
    default=None,
    help="Path of the output directory (will be create if it doesn't exist)"
  )

  return parser

