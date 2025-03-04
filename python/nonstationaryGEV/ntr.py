#%%
import utide
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

#%% FUNCTIONS
# Remove specific stations by record_id
def remove_stations(rsl):
    return rsl.sel(record_id=~rsl.record_id.isin([547, 548, 14]))

# Function to apply Butterworth lowpass filter
def butterworth_lowpass_xr(data, cutoff, fs, order=4):
    """
    Apply a Butterworth lowpass filter to the data along the time dimension,
    handling NaNs by interpolating temporarily.
    
    Parameters:
        data (xarray.DataArray): The data to be filtered (non-tidal residuals).
        cutoff (float): The cutoff frequency of the filter (in Hz).
        fs (float): The sampling frequency (in Hz).
        order (int): The order of the filter.

    Returns:
        xarray.DataArray: Filtered data with NaNs restored where they were in the original.
    """
    # Design the Butterworth filter
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Interpolate to fill NaNs temporarily, if any
    data_interpolated = data.interpolate_na(dim='time', method='linear')
    
    # Apply the filter along the time dimension
    filtered_values = filtfilt(b, a, data_interpolated.values, axis=data.get_axis_num('time'))
    
    # Recreate the filtered DataArray with the same coordinates and dimensions
    filtered_data = xr.DataArray(filtered_values, coords=data.coords, dims=data.dims)
    
    # Restore the original NaNs
    filtered_data = filtered_data.where(~data.isnull())
    
    return filtered_data

def bootstrap_percentile_with_ci(data, percentile=95, n_bootstrap=1000, conf_level=0.95):
    """
    Bootstrap the estimate of a specified percentile and compute confidence intervals.
    
    Parameters:
        data (xarray.DataArray): The daily data to process.
        percentile (float): Percentile to estimate (default is 95).
        n_bootstrap (int): Number of bootstrap samples.
        conf_level (float): Confidence level for CI (e.g., 0.95).
        
    Returns:
        xr.DataArray: Estimated percentile.
        xr.DataArray: Lower CI bound.
        xr.DataArray: Upper CI bound.
    """
    # Initialize lists to store results
    percentiles = []
    lower_ci = []
    upper_ci = []
    
    # Group by day of the year to apply bootstrapping to each day group
    for day, group in data.groupby('time.dayofyear'):
        # Extract values for this day
        clean_data = group.values

        # Perform bootstrapping on each day's data
        bootstrap_samples = [
            np.nanpercentile(np.random.choice(clean_data, size=len(clean_data), replace=True), percentile)
            for _ in range(n_bootstrap)
        ]
        
        # Calculate the percentile estimate and confidence intervals
        estimate = np.mean(bootstrap_samples)
        lower_bound = np.percentile(bootstrap_samples, (1 - conf_level) / 2 * 100)
        upper_bound = np.percentile(bootstrap_samples, (1 + conf_level) / 2 * 100)
        
        # Append results
        percentiles.append(estimate)
        lower_ci.append(lower_bound)
        upper_ci.append(upper_bound)

    # Define fractional year with 366 days for leap year handling
    fractional_year = np.linspace(0, 1, 366, endpoint=False)

    # Convert lists to DataArrays
    percentile_estimate = xr.DataArray(percentiles, coords={'fractional_year': fractional_year}, dims="fractional_year")
    lower_ci_bound = xr.DataArray(lower_ci, coords={'fractional_year': fractional_year}, dims="fractional_year")
    upper_ci_bound = xr.DataArray(upper_ci, coords={'fractional_year': fractional_year}, dims="fractional_year")

    return percentile_estimate, lower_ci_bound, upper_ci_bound

#%%

# Load dataset and remove specified stations
with xr.open_dataset('/Users/jfiedler/Documents/Repositories/SL_Hawaii/SL_Hawaii/data/rsl_hawaii.nc') as rsl:
    rsl = remove_stations(rsl)

recordID = 57   

# add yearday to rsl.coords
rsl['yearday'] = rsl.time.dt.dayofyear
start_year = rsl.time.dt.year.min()
fractional_year = 0 + (rsl.time.dt.dayofyear - 1) / 366
rsl['fractional_year'] = fractional_year


    # # Loop through each unique record_id in the dataset
    # for recordID in rsl.record_id.values:
        # Select sea level data for current record_id
sea_level =  rsl.sea_level.sel(record_id=recordID)

# select only data from 1993-2023
sea_level = sea_level.sel(time=slice('1993-01-01', '2023-12-31'))

# Convert time to decimal days since the start of the time series
tday = (sea_level.time - sea_level.time[0]).dt.total_seconds() / (3600 * 24)
#%% 
# STEP 1: Solve for tidal constituents       
coef = utide.solve(
    sea_level.time.values,
    sea_level.values,
    lat=rsl.lat.sel(record_id=recordID).values,
    nodal=True,
    trend=True,
    method='ols',
    conf_int='linear'
)


# trend = coef['slope'] * tday
trend = coef['mean'] + coef['slope'] * tday

#%%
# sea_level_detrended = sea_level - trend + rsl.MHHW.sel(record_id=recordID)
sea_level_detrended = sea_level - coef['slope'] * tday 
# sea_level_detrended = sea_level - coef['slope'] * tday - rsl.MHHW.sel(record_id=recordID)

#%%
# Reconstruct the tidal prediction
tide_pred = utide.reconstruct(sea_level.time.values, coef)


# Calculate daily high tide values
time = sea_level.time.values
tide_pred_da = xr.DataArray(tide_pred['h'], coords=[time], dims=['time']) - coef['slope'] * tday

#%%
# Calculate max tide for each day
daily_max_tide = tide_pred_da.resample(time='1D').max() - rsl.MHHW.sel(record_id=recordID)

# Plot the new time series of daily maximum tide levels
daily_max_tide.plot()

#%%
# # Step 1: Calculate the 95th percentile of daily max tide by day of year
# tide_pred_95 = daily_max_tide.groupby('time.dayofyear').reduce(np.nanpercentile, q=95)
tide_pred_95_estimate, tide_pred_95_lower_ci, tide_pred_95_upper_ci = bootstrap_percentile_with_ci(daily_max_tide)

#%%
# interpolate to align with rsl.time
tide_pred_95_estimate = tide_pred_95_estimate.interp(fractional_year=rsl.fractional_year)
tide_pred_95_lower_ci = tide_pred_95_lower_ci.interp(fractional_year=rsl.fractional_year)
tide_pred_95_upper_ci = tide_pred_95_upper_ci.interp(fractional_year=rsl.fractional_year)


# smooth the estimated 95th percentile with a 30-day moving average
tide_pred_95_estimate = tide_pred_95_estimate.rolling(time=30*24, center=True).mean()
tide_pred_95_lower_ci = tide_pred_95_lower_ci.rolling(time=30*24, center=True).mean()
tide_pred_95_upper_ci = tide_pred_95_upper_ci.rolling(time=30*24, center=True).mean()


# Plot the estimated 95th percentile with confidence intervals
tide_pred_95_estimate.sel(time = slice('1993-01-01','1993-12-30')).plot(label="95th Percentile")
tide_pred_95_lower_ci.sel(time = slice('1993-01-01','1993-12-30')).plot(label="Lower 95% CI", linestyle="--")
tide_pred_95_upper_ci.sel(time = slice('1993-01-01','1993-12-30')).plot(label="Upper 95% CI", linestyle="--")
plt.legend()
plt.show()

#%%
sea_level_detided = sea_level_detrended - tide_pred_da
sea_level_detided.plot()

#%%
# Sampling frequency (fs) in Hz (hourly data -> 1/3600 Hz)
fs = 1 / 3600

# Set a cutoff frequency for approximately monthly (30-day) cycles
# 1/2592000 Hz corresponds to roughly a 30-day period
cutoff = 1 / (30 * 24 * 3600)

# Apply the Butterworth filter to extract the seasonal component
seasonal = butterworth_lowpass_xr(sea_level_detided, cutoff, fs)

# Now group by day of the year to compute climatological seasonal cycle
seasonal_monthly = seasonal.groupby('time.dayofyear').mean(skipna=True)
# Interpolate for leap years
seasonal_monthly = seasonal_monthly.interp(dayofyear=np.arange(1, 366), method="linear")
# Interpolate the seasonal component to align with yearday coordinate
seasonal_interp = seasonal_monthly.interp(dayofyear=rsl.yearday)

seasonal_interp.plot()

#%%
# Short-term is the remainder after removing the tide and seasonal components
short_term = sea_level_detided - seasonal 

# Extract daily max short-term residuals
short_term_daily_max = short_term.resample(time='1D').max()

short_term_95_estimate, short_term_95_lower_ci, short_term_95_upper_ci = bootstrap_percentile_with_ci(short_term_daily_max)

# interpolate to align with rsl.time
short_term_95_estimate = short_term_95_estimate.interp(fractional_year=rsl.fractional_year)
short_term_95_lower_ci = short_term_95_lower_ci.interp(fractional_year=rsl.fractional_year)
short_term_95_upper_ci = short_term_95_upper_ci.interp(fractional_year=rsl.fractional_year)

# smooth the estimated 95th percentile with a 30-day moving average
short_term_95_estimate = short_term_95_estimate.rolling(time=30*24, center=True).median()
short_term_95_lower_ci = short_term_95_lower_ci.rolling(time=30*24, center=True).median()
short_term_95_upper_ci = short_term_95_upper_ci.rolling(time=30*24, center=True).median()

#%%
total = tide_pred_95_estimate + seasonal_interp + short_term_95_estimate

# generate confidence intervals for total
total_lower_ci = tide_pred_95_lower_ci + seasonal_interp + short_term_95_lower_ci
total_upper_ci = tide_pred_95_upper_ci + seasonal_interp + short_term_95_upper_ci
#%%
from matplotlib import dates as mdates
import pandas as pd
# Convert each dataset to meters by dividing by 1000
tide_pred_95_estimate_m = tide_pred_95_estimate / 1000
tide_pred_95_lower_ci_m = tide_pred_95_lower_ci / 1000
tide_pred_95_upper_ci_m = tide_pred_95_upper_ci / 1000

seasonal_interp_m = seasonal_interp / 1000

short_term_95_estimate_m = short_term_95_estimate / 1000
short_term_95_lower_ci_m = short_term_95_lower_ci / 1000
short_term_95_upper_ci_m = short_term_95_upper_ci / 1000

total_m = total / 1000
total_lower_ci_m = total_lower_ci / 1000
total_upper_ci_m = total_upper_ci / 1000



fig, ax = plt.subplots()

tide_pred_95_estimate_m.sel(time = slice('1993-01-01','1993-12-30')).plot(label='tide (95%)')

# fill between the confidence intervals
# ax.fill_between(tide_pred_95_estimate_m.sel(time = slice('1993-01-01','1993-12-30')).time.values, 
#                  tide_pred_95_lower_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
#                  tide_pred_95_upper_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
#                  alpha=0.3)



seasonal_interp_m.sel(time = slice('1993-01-01','1993-12-30')).plot(label='seasonal', color='green')


short_term_95_estimate_m.sel(time = slice('1993-01-01','1993-12-30')).plot(label='short-term (95%)', color='red')
# fill between the confidence intervals 
ax.fill_between(short_term_95_estimate_m.sel(time = slice('1993-01-01','1993-12-30')).time.values, 
                 short_term_95_lower_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
                 short_term_95_upper_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
                 alpha=0.15, color='red')

total_m.sel(time = slice('1993-01-01','1993-12-30')).plot(label='total', color='black')

# fill between the confidence intervals
ax.fill_between(total_m.sel(time = slice('1993-01-01','1993-12-30')).time.values, 
                 total_lower_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
                 total_upper_ci_m.sel(time = slice('1993-01-01','1993-12-30')).values, 
                 alpha=0.2, color='black')

title = f'Station {recordID} - {rsl.station_name.sel(record_id=recordID).item()}'

ax.set_title('High water level climatology \n ' + title + ' (1993-2023)')
ax.legend()
ax.set_ylabel('Sea Level (m, MHHW)')


# Format x-axis to show only month names
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Set major ticks to each month
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Show month abbreviation

# remove white space before january and after december
ax.set_xlim(pd.Timestamp('1993-01-01'), pd.Timestamp('1993-12-31'))

# add extremes to plot from /Users/jfiedler/Documents/Repositories/SL_Hawaii/SL_Hawaii/output/SL_top_10_table_Honolulu, Hawaii.csv
extremes = pd.read_csv('/Users/jfiedler/Documents/Repositories/SL_Hawaii/SL_Hawaii/output/SL_top_10_table_Honolulu, Hawaii.csv')
#extract only the highest values (columns highest, highest date, highest ONI Mode)
extremes = extremes[['Highest','Highest Date','Highest ONI Mode']]
# convert date to datetime
extremes['Highest Date'] = pd.to_datetime(extremes['Highest Date'])
# set date as index
extremes.set_index('Highest Date', inplace=True)



# get yearday of extremes
extremes['yearday'] = extremes.index.dayofyear

# convert this to the day in 1993
extremes['PlotDate'] = pd.to_datetime('1993-01-01') + pd.to_timedelta(extremes['yearday'], unit='D')

# plot extremes and color by ONI mode
oni_mode = extremes['Highest ONI Mode']
colors = oni_mode.map({'La Nina': 'blue', 'Neutral': 'green', 'El Nino': 'red'})
ax.scatter(extremes['PlotDate'], extremes['Highest'], color=colors, label='extremes')

