#%%
from models import run_all_models
from imports import *
from plotting import plotExtremeSeasonality, plotTimeDependentReturnValue
from helpers import make_directories

# print the current working directory
cwd = os.getcwd()
# go up two levels
base_dir = os.path.abspath(os.path.join(cwd, "../.."))

dirs = make_directoryDict(base_dir)


recordID = 57
runWithoutModel = True
returnPeriod = [2,10,50,100]
year0plot = 1993
saveToFile = False
climateIndex = ['A0','AAO','BEST','DMI','ONI','PDO','PMM','PNA','TNA']

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
SignifCvte = np.zeros(len(climateIndex))

for i in np.arange(0, len(climateIndex)):
    # Run the model with each climate index
    STNDtoMHHW, station_name, year0, mm, x_s, w_s, w_cvte2, w_T = run_all_models(
        rsl_hourly, recordID, runWithoutModel, dirs, ReturnPeriod=returnPeriod, CIname=climateIndex[i]
    )
    
    # Calculate the difference between model coefficients
    diffe = w_cvte2[0] - w_T[0]
    
    # Degrees of freedom (2 for location, 2 for scale?)
    p = 4  # Modify based on the parameters being compared
    
    # Compute the significance using the chi-squared cumulative distribution
    SignifCvte[i] = chi2.cdf(2 * diffe, p)



#%%

# plot seasonal cycle
figSeasonal, cmap = plotExtremeSeasonality(mm['t'],mm['monthly_max'],x_s,w_s,recordID, STNDtoMHHW, dirs, station_name, ReturnPeriod=returnPeriod,SampleRate=12,saveToFile=saveToFile)
#%%
# plot time series
figTimeSeries = plotTimeDependentReturnValue(str(recordID), STNDtoMHHW, dirs['model_output_dir'], station_name, dirs['output_dir'], mm, year0plot, saveToFile=saveToFile)



figSeasonal.show()
figTimeSeries.show()
# %%
