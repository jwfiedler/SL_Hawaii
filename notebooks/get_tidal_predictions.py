#%%
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d
import utide
from pathlib import Path
from joblib import Parallel, delayed
from scipy import stats

# Load data
data_dir = Path('../data')
rsl_hourly = xr.open_dataset(data_dir / 'rsl_hawaii.nc')
#import rsl_daily
rsl_daily_all = xr.open_dataset(data_dir/ 'rsl_daily_hawaii.nc')

#make rsl_daily a subset - do not include TGs with more than 25% NaNs
data_coverage = rsl_daily_all['rsl_mhhw'].count(dim='time')/len(rsl_daily_all.time)

#drop all locations with data_coverage less than 80%
rsl_daily = rsl_daily_all.where(data_coverage>0.80,drop=True)

#include only the TGs that are in rsl_subset
rsl_hourly = rsl_hourly.sel(record_id = rsl_daily.record_id.values)

#%%
def process_trend_with_nan(sea_level_anomaly, weighted=False):
    # Flatten the data and get a time index
    sea_level_anomaly = sea_level_anomaly.transpose('time', ...)
    sla_flat = sea_level_anomaly.values.reshape(sea_level_anomaly.shape[0], -1)
    time_index = pd.to_datetime(sea_level_anomaly.time.values).to_julian_date()

    detrended_flat = np.full_like(sla_flat, fill_value=np.nan)

    # Loop over each grid point
    for i in range(sla_flat.shape[1]):
        y = sla_flat[:, i]
        mask = ~np.isnan(y)

        if np.any(mask):
            time_masked = time_index[mask]
            y_masked = y[mask]

            slope, intercept, _, _, _ = stats.linregress(time_masked, y_masked)
            trend = slope * time_index + intercept

            detrended_flat[:, i] = y - trend

    detrended = detrended_flat.reshape(sea_level_anomaly.shape)

    # Calculate trend magnitude
    sea_level_trend = sea_level_anomaly - detrended
    trend_mag = sea_level_trend[-1] - sea_level_trend[0]

    times = pd.to_datetime(sea_level_anomaly['time'].values)
    time_mag = (times[-1] - times[0]).days / 365.25  # in years

    trend_rate = trend_mag / time_mag

    if weighted:
        weights = np.cos(np.deg2rad(sea_level_anomaly.latitude))
        weights.name = 'weights'

        trend_mag = (trend_mag * weights).mean()
        trend_rate = (trend_rate * weights).mean()
        sea_level_trend = (sea_level_trend * weights).mean(dim=['latitude', 'longitude'])

    return trend_mag, sea_level_trend, trend_rate

#%%

# Calculate the trend for each station
trend_mag, sea_level_trend, trend_rate = process_trend_with_nan(rsl_daily['rsl_anomaly'], weighted=False)

#%%

# Helper function to split data into continuous segments
def split_continuous_segments(time, data, max_gap_hours=12):
    df = pd.DataFrame({'time': time, 'data': data})
    df['time_diff'] = df['time'].diff().dt.total_seconds() / 3600
    df['segment'] = (df['time_diff'] > max_gap_hours).cumsum()
    segments = [list(zip(group['time'], group['data'])) for _, group in df.groupby('segment')]
    return segments

#%%
# Function to process each station
def process_station(station_id):
    print(f'Processing {rsl_hourly["station_name"].sel(record_id=station_id).values}')
    sea_level_data = rsl_hourly['sea_level'].sel(record_id=station_id).values
    sea_level_data -= trend_rate.sel(record_id=station_id).values * 1000 * rsl_hourly['time'].values.astype('datetime64[s]').astype(np.float64)
    print(f'Percentage of NaNs in the data: {np.isnan(sea_level_data).sum() / len(sea_level_data) * 100:.2f}%')

    latitude = rsl_hourly['lat'].sel(record_id=station_id).values
    time_series = pd.to_datetime(rsl_hourly['time'].values)
    segments = split_continuous_segments(time_series, sea_level_data)
    print(f'Number of segments: {len(segments)}')

    tidal_pred = np.full_like(sea_level_data, np.nan)

    for segment_idx, segment in enumerate(segments, 1):
        print(f'Processing segment {segment_idx} of {len(segments)}')
        segment_time, segment_data = zip(*segment)
        segment_time = np.array(segment_time)
        segment_data = np.array(segment_data)

        segment_time_numeric = segment_time.astype('datetime64[s]').astype(np.float64)

        if np.isnan(segment_data).any():
            mask = np.isnan(segment_data)
            interp_func = interp1d(segment_time_numeric[~mask], segment_data[~mask], kind='linear', fill_value="extrapolate")
            segment_data[mask] = interp_func(segment_time_numeric[mask])

        coef = utide.solve(segment_time, segment_data, lat=latitude)
        tide_pred = utide.reconstruct(segment_time, coef)

        idxs = np.searchsorted(time_series, segment_time)
        tidal_pred[idxs] = tide_pred.h

    return station_id, tidal_pred

#%%
# Main script
if not (data_dir / 'rsl_hawaii_tidal_predictions.nc').exists():
    print('rsl_hawaii_tidal_predictions.nc not found. Will proceed with tidal analysis and prediction.')

    sea_level_shape = rsl_hourly['sea_level'].shape
    time_data = rsl_hourly['time'].values
    
    # Parallel processing for each station
    results = Parallel(n_jobs=-1)(delayed(process_station)(station_id) for station_id in rsl_hourly['record_id'].values)

    tidal_predictions = np.full(sea_level_shape, np.nan)
    for station_id, pred in results:
        station_idx = np.where(rsl_hourly['record_id'].values == station_id)[0][0]
        tidal_predictions[station_idx] = pred
    tidal_predictions_ds = xr.Dataset(
        data_vars={'tidal_prediction': (('record_id', 'time'), tidal_predictions)},
        coords={'time': rsl_hourly['time'].values, 'record_id': rsl_hourly['record_id'].values}
    )
    tidal_predictions_ds.to_netcdf(data_dir / 'rsl_hawaii_tidal_predictions.nc')
    print(f'Tidal predictions saved to: {data_dir / "rsl_hawaii_tidal_predictions.nc"}')
else:
    print('rsl_hawaii_tidal_predictions.nc found. Proceed.')
# %%