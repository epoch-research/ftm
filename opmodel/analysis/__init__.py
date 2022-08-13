import os
import sys
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from ..core.opmodel import SimulateTakeOff
from ..core.utils import display, log, draw_oom_lines
from ..core.utils import *
from ..report.report import Report

class ArgumentParserWrapper(argparse.ArgumentParser):
  def __init__(self):
    self.abbrev_to_full_name = {}
    super().__init__()

  def add_argument(self, *args, **kwargs):
    if len(args) >= 2:
      abbr = args[0][1:] # get rid of '-'
      full_name = args[1][2:] # get rid of '--'
      self.abbrev_to_full_name[abbr] = full_name

    super().add_argument(*args, **kwargs)

# Some constants
def init_cli_arguments():
  parser = ArgumentParserWrapper()

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

  parser.add_argument(
    "--t-start",
    type=int,
    default=None,
  )

  parser.add_argument(
    "--t-end",
    type=int,
    default=None,
  )

  parser.add_argument(
    "--t-step",
    type=float,
    default=None,
  )

  parser.add_argument(
    "-p",
    "--param-table-url",
    default=None,
  )

  parser.add_argument(
    "--ajeya-dist-url",
    default=None,
  )

  return parser

def handle_cli_arguments(parser):
  # Load arguments from config files (if any)
  caller = inspect.stack()[1]
  module_name = inspect.getmodulename(caller.filename)
  options = get_option(module_name)
  if options:
    clean_options = {}
    for k, v in options.items():
      if k in parser.abbrev_to_full_name:
        k = parser.abbrev_to_full_name[k]
      k = k.replace('-', '_')
      clean_options[k] = v
    parser.set_defaults(**clean_options)

  args = parser.parse_args()

  if args.workbook_url is not None:
    set_input_workbook(args.workbook_url)

  if args.param_table_url is not None:
    set_parameter_table_url(args.param_table_url)

  if args.ajeya_dist_url is not None:
    set_ajeya_dist_url(args.ajeya_dist_url)

  if args.t_start is not None: set_option('t_start', args.t_start)
  if args.t_end is not None: set_option('t_end', args.t_end)
  if args.t_step is not None: set_option('t_step', args.t_step)

  return args
