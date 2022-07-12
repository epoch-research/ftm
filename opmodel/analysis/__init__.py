import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display, log, get_parameter_table
from ..report.report import Report

# Some constants
CACHE_DIR = '_cache_/diffy'
MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(MODULE_DIR, '..', '..', CACHE_DIR))

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

