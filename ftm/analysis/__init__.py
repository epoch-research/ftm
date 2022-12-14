import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

from ..core.model import SimulateTakeOff
from ..core.utils import *
from .report import Report

def inputs_nan_format(row, col, index_r, index_c, cell):
  return ''
