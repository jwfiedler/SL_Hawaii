from pathlib import Path
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def get_stationinfo(data_dir):
    # Use a context manager to ensure the dataset is closed properly
    with xr.load_dataset(data_dir/ 'rsl_daily_hawaii.nc') as rsl:
        # Convert relevant data from xarray to a pandas DataFrame
        station_info = rsl[['lat', 'lon', 'station_name', 'record_id']].to_dataframe().reset_index()

        # Convert station_name to string and remove everything after the comma
        station_info['station_name'] = station_info['station_name'].astype(str).str.split(',').str[0]

        # Add default values for offsetting and text alignment
        station_info['offsetlon'] = 0.2
        station_info['offsetlat'] = 0.2
        station_info['ha'] = 'left'
        station_info['fontcolor'] = 'gray'

        # Define custom offsets for specific stations using a dictionary
        custom_offsets = {
            'Nawiliwili': {'offsetlat': 0.4},
            'Mokuoloe': {'offsetlat': 0.4},
            'Hilo': {'offsetlat': 0.3, 'offsetlon': 0.3},
            'Kawaihae': {'offsetlat': -0.5, 'offsetlon': -0.5, 'ha': 'right'},
            'Kaumalapau': {'offsetlat': -0.6, 'offsetlon': 0, 'ha': 'right'},
            'Barbers Point': {'offsetlat': -0.2, 'offsetlon': -0.5, 'ha': 'right'},
            'Honolulu': {'offsetlat': -0.6, 'offsetlon': -0.1, 'ha': 'right'}
        }

        # Apply the custom offsets to the DataFrame
        for station, settings in custom_offsets.items():
            for column, value in settings.items():
                station_info.loc[station_info['station_name'] == station, column] = value

    # Dataset will be closed automatically when the `with` block exits
    return station_info


def plot_thin_map_hawaii(ax):
    # Add features to the map
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    # Set the extent to focus on Hawaii and surrounding areas
    ax.set_extent([-179, -153, 15, 30])  # Adjust to focus on Hawaii


def plot_station_labels(ax, station_info):# Add labels to the stations
    
    station_label = {}
    for i, row in station_info.iterrows():
        ax.scatter(row['lon'], row['lat'], color='black', s=10, transform=ccrs.PlateCarree())
        station_label[i] = ax.text(row['lon'] + row['offsetlon'], row['lat'] + row['offsetlat'], row['station_name'],
                                   ha=row['ha'], va='center', transform=ccrs.PlateCarree(), fontsize=8, color=row['fontcolor'])

    
    return station_label


def get_top_ten(rsl, rid, mode='max'):
    # Convert data to a pandas Series
    sea_level_series = rsl.sea_level.isel(record_id=rid).to_series()


   # Select top 100 values based on the mode
    if mode == 'max':
        top_values = sea_level_series.nlargest(100)
    elif mode == 'min':
        top_values = sea_level_series.nsmallest(100)
    else:
        raise ValueError('mode must be either "max" or "min"')

    # Filter to find unique events spaced by at least 3 days
    filtered_dates = []
    top_10_values = pd.Series()

    for date, value in top_values.items():
        if all(abs((date - pd.to_datetime(added_date)).days) > 3 for added_date in filtered_dates):
            filtered_dates.append(date)
            top_10_values[date] = value
        if len(filtered_dates) == 10:
            break
    rank = np.arange(1,11)

    station_name = str(rsl['station_name'].isel(record_id=rid).values)
    record_id = str(rsl['record_id'].isel(record_id=rid).values)  

    top_10_values = pd.DataFrame({'rank': rank, 'date': top_10_values.index, 'sea level (m)': top_10_values.values})  
    top_10_values['station_name'] = station_name
    top_10_values['record_id'] = record_id
    top_10_values['type'] = mode

    #round the date to the nearest hour
    top_10_values['date'] = top_10_values['date'].dt.round('h')

    return top_10_values


def get_top_10_table(rsl,rid,ONI_dir):
    # make a table of the top 10 values, sorted by size and with date
    top_10_values_max = get_top_ten(rsl, rid, mode='max')
    top_10_values_min = get_top_ten(rsl, rid, mode='min')

    top_10_table = pd.concat([top_10_values_max,top_10_values_min])

    # cross reference the dates with the oni data to see if they are during an El Nino or La Nina event
    oni = pd.read_csv(ONI_dir / 'oni.csv', index_col='time', parse_dates=True)

    # El Nino is true when ONI > 0.5 for 5 consecutive periods 
    oni['El Nino'] = (oni['ONI'] > 0.5).rolling(window=5).sum() == 5

    # La Nina is true when ONI < -0.5 for 5 consecutive periods 
    oni['La Nina'] = (oni['ONI'] < -0.5).rolling(window=5).sum() == 5

    # add a new column to oni_min called mode, where mode is either 'El Nino', 'La Nina', or 'Neutral'
    oni['ONI Mode'] = 'Neutral'
    oni.loc[oni['La Nina'] ==True, 'ONI Mode'] = 'La Nina'
    oni.loc[oni['El Nino'] ==True, 'ONI Mode'] = 'El Nino'

    #drop the La Nina and El Nino columns
    oni = oni.drop(columns=['La Nina', 'El Nino'])

    #Extract ONI values for the corresponding dates of top_10_table
    oni_val = oni.reindex(top_10_table['date'], method='nearest')
    
    # add the oni values to the top_10_table
    top_10_table = pd.merge(top_10_table, oni_val, left_on='date', right_index=True)

    return top_10_table
