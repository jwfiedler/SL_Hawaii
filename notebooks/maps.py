from pathlib import Path
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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