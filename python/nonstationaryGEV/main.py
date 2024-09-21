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


recordID =57
runWithoutModel = False
returnPeriod = [2,10,50,100]
year0plot = 1993
saveToFile = True
climateIndex = ['AO','AAO','BEST','DMI','ONI','PDO','PMM','PNA','TNA']

#%%
# get dataset of hourly sea level data
rsl = xr.open_dataset(dirs['data_dir']/ 'rsl_hawaii.nc')

# remove stations 547,548, 14
rsl_hourly = rsl.sel(record_id=~rsl.record_id.isin([547,548,14]))

# close the file
rsl.close()

make_directories(rsl_hourly,dirs)
#%%
# Preallocate the significance array
SignifCvte1 = np.zeros(len(climateIndex))
SignifCvte2_loc = np.zeros(len(climateIndex))
SignifCvte2_T = np.zeros(len(climateIndex))

runWithoutModel=True
_, _, _, _, _, _, x_N, w_N, wcomp, SignifN = run_noClimateIndex_models(rsl_hourly,recordID,runWithoutModel,dirs, returnPeriod, CIname='None')
STNDtoMHHW, station_name, year0, mm, ampCvte1, SignifCvte1 = run_CI_models(rsl_hourly,recordID,False,dirs, returnPeriod, climateIndex,x_N, w_N, wcomp, SignifN)

#%%
# Initialize an empty list to store results
results = []

for i in np.arange(0, len(climateIndex)):
    covariate_params = f'cvte_location_params_{climateIndex[i]}.json'
    
    # Create the full path for the JSON file
    jsonpath = Path(dirs['model_output_dir']) / str(recordID) / covariate_params

    # Open and read the JSON file
    with open(jsonpath, 'r') as f:
        output = json.load(f)
        w, mio, standard_error = (np.array(output[key]) for key in ['w', 'mio', 'standard_error'])

    # Store the results in a list
    results.append({
        'Climate Index': climateIndex[i],
        'Amplitude of CI param': w[-1],  
        'Standard Error of CI param': standard_error[-1]
    })

# Convert the results list to a DataFrame
df_cvteLocation = pd.DataFrame(results)

# add Significance to the dataframe
df_cvteLocation['Significance (over trend)'] = SignifCvte1

df_cvteLocation

#%%
# Initialize an empty list to store results
results = []

for i in np.arange(0, len(climateIndex)):
    covariate_params = f'cvte_scale_params_{climateIndex[i]}.json'
    
    # Create the full path for the JSON file
    jsonpath = Path(dirs['model_output_dir']) / str(recordID) / covariate_params

    # Open and read the JSON file
    with open(jsonpath, 'r') as f:
        output = json.load(f)
        w, mio, standard_error = (np.array(output[key]) for key in ['w', 'mio', 'standard_error'])

    if standard_error[-1] == 0:
        standard_error[-1] = np.nan


    # Store the results in a list
    results.append({
        'Climate Index': climateIndex[i],
        'Amplitude of CI param': w[-1],  
        'Standard Error of CI param': standard_error[-1]
    })

# Convert the results list to a DataFrame
df_cvteScale = pd.DataFrame(results)

# add Significance to the dataframe
df_cvteScale['Significance over cvte_loc'] = SignifCvte2_loc
df_cvteScale['Significance over trend'] = SignifCvte2_T

df_cvteScale

#%%

# # plot seasonal cycle
# figSeasonal, cmap = plotExtremeSeasonality(mm['t'],mm['monthly_max'],x_s,w_s,recordID, STNDtoMHHW, dirs, station_name, ReturnPeriod=returnPeriod,SampleRate=12,saveToFile=saveToFile)
# #%%
# # plot time series
# figTimeSeries = plotTimeDependentReturnValue(str(recordID), STNDtoMHHW, dirs['model_output_dir'], station_name, dirs['output_dir'], mm, year0plot, saveToFile=saveToFile)



# figSeasonal.show()
# figTimeSeries.show()
# # %%
