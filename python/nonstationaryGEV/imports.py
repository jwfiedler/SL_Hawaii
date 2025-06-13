import pandas as pd
import numpy as np
import scipy.io
from datetime import datetime, timedelta
from scipy.stats import chi2
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import subprocess
import json
import os

from pathlib import Path
import xarray as xr
from scipy.stats import chi2

import seaborn as sns
import matplotlib.colors as mcolors
import plotly.graph_objects as go

#set up directories as environment variables
os.environ["DATA_DIR"] = "~/Documents/SL_Hawaii_data/data"
os.environ["OUTPUT_DIR"] = "~/Documents/SL_Hawaii_data/output"

#set up directories as Path objects
data_dir = Path(os.environ["DATA_DIR"]).expanduser()
output_dir = Path(os.environ["OUTPUT_DIR"]).expanduser()


param_names = ['Annual seasonal cycle',
    'Semiannual seasonal cycle',
    'Triannual seasonal cycle',
    'Long-term Trend in Location',
    'Covariate in Location',
    'Covariate in Scale',
    'Nodal Cycle',
    'Perigean Cycle']


def make_directoryDict(base_dir):

    base_dir = Path(base_dir)


    dirs = {
        'data_dir': data_dir, 
        'output_dir': output_dir / 'extremes',
        'input_dir': base_dir / 'model_input',
        'matrix_dir': base_dir / 'matrix',
        'model_output_dir': data_dir / 'GEV_model_output',
        'CI_dir': data_dir / 'climate_indices',
        'run_dir': data_dir / 'model_run'
    }
    
    return dirs


def define_limits(run_dir):
    limits = [
        (0.01, 7.5),
        (0.001, 2.5),
        (-0.35, 0.15),
        (-2.001, 2.001),
        (-2.001, 2.001),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.25, 0.25),
        (-0.25, 0.25),
        (-0.3, 0.3),
        (-0.5, 0.5),
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, 0.2),
        (-0.2, 0.2)
    ]

    limitsPath = run_dir / 'limits.txt'

    with open(limitsPath, "w") as file:
        for row in limits:
            file.write(f"{row[0]} {row[1]}\n")

    return limitsPath