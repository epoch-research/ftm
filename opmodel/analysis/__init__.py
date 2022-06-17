import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display
from ..report.report import Report

def init_cli_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "-o",
    "--output-file",
    default="exploration_analysis.html",
    help="Path of the output report (absolute or relative to the report directory)"
  )

  parser.add_argument(
    "-d",
    "--output-dir",
    default=Report.default_report_path(),
    help="Path of the output directory (will be create if it doesn't exist)"
  )

  return parser

class Log:
  INFO_LEVEL = 1

  def __init__(self, level=0):
    self.level = level
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

log = Log(level=Log.INFO_LEVEL)

