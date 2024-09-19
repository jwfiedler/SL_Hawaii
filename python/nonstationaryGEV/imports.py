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


param_names = ['Annual seasonal cycle',
    'Semiannual seasonal cycle',
    'Triannual seasonal cycle',
    'Long-term Trend in Location',
    'Covariate in Location',
    'Covariate in Scale',
    'Nodal Cycle']



def make_directoryDict(base_dir):

    base_dir = Path(base_dir)
    base_data_dir = Path(base_dir / 'data')

    dirs = {
        'data_dir': base_data_dir, 
        'output_dir': base_dir / 'output/extremes',
        'input_dir': base_dir / 'model_input',
        'matrix_dir': base_dir / 'matrix',
        'model_output_dir': base_data_dir / 'GEV_model_output',
        'CI_dir': base_data_dir / 'climate_indices',
        'run_dir': base_data_dir / 'model_run'
    }
    
    return dirs


