import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import netCDF4 as nc
from scipy.interpolate import griddata
import earthaccess

# Load environment variables
load_dotenv()  

username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

earthaccess.login(strategy="environment")

# Define datacube config (must match with trained model settings)
DATACUBE_CONFIG = {
    'spatial_extent_km': 30,
    'temporal_extent_days': 5,
    'spatial_resolution_km': 2,
}
MODALITY = 'chlor_a'

def search_modis_l2_data(date, spatial_bounds):
    """Searches for MODIS L2 data for a specific date and bounding box."""
    if not earthaccess: return []
    bbox = (
        spatial_bounds['lon_min'], spatial_bounds['lat_min'],
        spatial_bounds['lon_max'], spatial_bounds['lat_max']
    )
    try:
        granules = earthaccess.search_data(
            short_name='MODISA_L2_OC',
            temporal=(date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d')),
            bounding_box=bbox,
            count=-1 # Get all granules for the day
        )
        return granules
    except Exception as e:
        print(f"Earthaccess search error for {date}: {e}")
        return []

def main():
    # Sample data for testing
    lat = 24.79667
    lon = -80.78388
    start_date_str = '2015-01-18'

    print(f"Fetching nc files ({lat}, {lon})")

    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    num_days = DATACUBE_CONFIG['temporal_extent_days']
    datacube_3d = np.zeros((grid_size, grid_size, num_days))

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD.")

    # Calculate spatial bounds for the API query
    extent_deg = DATACUBE_CONFIG['spatial_extent_km'] / 111.0
    spatial_bounds = {
        'lat_min': lat - extent_deg / 2, 'lat_max': lat + extent_deg / 2,
        'lon_min': lon - extent_deg / 2, 'lon_max': lon + extent_deg / 2
    }

    for day_offset in range(num_days):
        target_date = start_date + timedelta(days=day_offset)
        print(f"Fetching data for Day {day_offset + 1}: {target_date.strftime('%Y-%m-%d')}...")

        granules = search_modis_l2_data(target_date, spatial_bounds)
        if not granules:
            print(f"No satellite data found for this day.")
            continue # Leave this day's slice as zeros

        # Download the found granules to a local cache
        raw_dir = Path("habnet_data_cache")
        raw_dir.mkdir(exist_ok=True)
        files = earthaccess.download(granules, local_path=str(raw_dir))

if __name__=='__main__':
    main()