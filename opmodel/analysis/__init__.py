import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display, log, draw_oom_lines
from ..core.utils import get_parameter_table, get_ajeya_dist, get_rank_correlations, set_omni_excel_url
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

  parser.add_argument(
    "-w",
    "--workbook-url",
    default=None,
    help="Url of the omni workbook"
  )

  return parser

def handle_cli_arguments(parser):
  args = parser.parse_args()

  if args.workbook_url is not None:
    set_omni_excel_url(args.workbook_url)
  del args.workbook_url

  return args
