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


#%% Test correlation and lag between climate index and sea level
recordID = 57
from helpers import get_monthly_max_time_series, get_covariate
CI_dir = dirs['CI_dir']

climateIndex = ['AO','AAO','BEST','DMI','ONI','PDO','PMM','PNA','TNA']

#%%
# get dataset of hourly sea level data
rsl = xr.open_dataset(dirs['data_dir']/ 'rsl_hawaii.nc')

# remove stations 547,548, 14
rsl_hourly = rsl.sel(record_id=~rsl.record_id.isin([547,548,14]))


# get dataset of monthly max sea level data
mm, STNDtoMHHW, station_name, year0 = get_monthly_max_time_series(recordID, rsl_hourly)
mmax = mm['monthly_max'].to_numpy()
CIcorr = np.zeros((len(climateIndex), 36))

for indCI in range(len(climateIndex)):
    CI = get_covariate(mm['t_monthly_max'], CI_dir, CIname = climateIndex[indCI])

    # Define the number of lags
    lag = 36

    # Initialize an array to store the correlation values for each lag
    corr = np.zeros(lag)

    # Calculate lagged correlation
    for i in range(1, lag + 1):
        # CI[0:-i] and mmax[i:] will both have length len(CI) - i
        corr[i - 1] = np.corrcoef(CI[:-i], mmax[i:])[0, 1]

    CIcorr[indCI, :] = corr
#%%
#plot CIcorr, the correlation between climate index and sea level for each climate index
fig, ax = plt.subplots()
ax.plot(np.arange(1, 37), CIcorr.T)
ax.set_xlabel('Lag (months)')
ax.set_ylabel('Correlation')
ax.set_title('Correlation between climate index and sea level')
ax.legend(climateIndex)
# %%
