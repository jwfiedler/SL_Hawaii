#%%
#%%
from models import run_CI_models
from models import run_noClimateIndex_models
from imports import *
from plotting import plotExtremeSeasonality, plotTimeDependentReturnValue
from helpers import make_directories
import sys



def in_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    return gettrace() is not None

def in_interactive_mode():
    try:
        get_ipython()
        return True
    except NameError:
        return False

# Determine base directory based on mode
cwd = os.getcwd()

if in_debug_mode():
    base_dir = cwd  # For debug mode
elif in_interactive_mode():
    base_dir = os.path.abspath(os.path.join(cwd, "../.."))  # For interactive mode (Jupyter/IPython)
else:
    base_dir = cwd  # For normal execution

print(f"Base directory: {base_dir}")
dirs = make_directoryDict(base_dir)

recordID = 57
runWithoutModel = False
returnPeriod = [2,10,50,100]
year0plot = 1993
saveToFile = True
climateIndex = ['BEST']
# %%

#%%

#load /Users/juliafiedler/Documents/Repositories/SL_Hawaii/SL_Hawaii/data/GEV_model_output/57/seasonal_params.json
# Load the seasonal parameters
with open(Path(dirs['model_output_dir']) / str(recordID) / 'seasonal_params.json', 'r') as f:
    output = json.load(f)
    w, mio, standard_error,x = (np.array(output[key]) for key in ['w', 'mio', 'standard_error','x'])


#%% Now open the old seasonal params file
with open(Path('/Users/juliafiedler/Library/CloudStorage/GoogleDrive-jfiedler@hawaii.edu/My Drive/NOAA/GEV_model_output_OLD') / str(recordID) / 'seasonal_params.json', 'r') as f:
    output = json.load(f)
    w2, mio2, standard_error2,x2 = (np.array(output[key]) for key in ['w', 'mio', 'standard_error','x'])