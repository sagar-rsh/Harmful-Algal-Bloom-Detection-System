import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import netCDF4 as nc
from scipy.interpolate import griddata
import earthaccess

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

def extract_modality_from_granule(file_path, spatial_bounds):
    """Extracts chlorophyll-a data from a single downloaded .nc file."""
    try:
        with nc.Dataset(file_path, 'r') as ds:
            if 'geophysical_data' not in ds.groups or 'navigation_data' not in ds.groups:
                return None
            
            geo_data = ds.groups['geophysical_data']
            nav_data = ds.groups['navigation_data']
            lats = nav_data.variables['latitude'][:]
            lons = nav_data.variables['longitude'][:]

            lat_mask = (lats >= spatial_bounds['lat_min']) & (lats <= spatial_bounds['lat_max'])
            lon_mask = (lons >= spatial_bounds['lon_min']) & (lons <= spatial_bounds['lon_max'])
            spatial_mask = lat_mask & lon_mask

            if not np.any(spatial_mask): 
                return None
            
            if MODALITY in geo_data.variables:
                mod_data = geo_data.variables[MODALITY][:]
                if mod_data.shape == lats.shape:
                    return {'lats': lats[spatial_mask], 'lons': lons[spatial_mask], 'values': mod_data[spatial_mask]}
    except Exception as e:
        print(f"Error reading granule {Path(file_path).name}: {e}")
    return None

def reproject_to_grid(lats, lons, values, spatial_bounds):
    """Reprojects scattered satellite data points onto a regular grid."""
    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    target_lats = np.linspace(spatial_bounds['lat_min'], spatial_bounds['lat_max'], grid_size)
    target_lons = np.linspace(spatial_bounds['lon_min'], spatial_bounds['lon_max'], grid_size)
    grid_lons, grid_lats = np.meshgrid(target_lons, target_lats)

    source_points = np.column_stack((lons.ravel(), lats.ravel()))
    target_points = np.column_stack((grid_lons.ravel(), grid_lats.ravel()))
    
    valid_mask = np.isfinite(values.ravel())
    if np.sum(valid_mask) < 4: return np.zeros((grid_size, grid_size))

    interpolated = griddata(
        source_points[valid_mask], values.ravel()[valid_mask],
        target_points, method='linear', fill_value=0.0
    )
    return interpolated.reshape((grid_size, grid_size))

def generate_prediction_datacube(lat, lon, start_date_str):
    """Generates a single 15x15x5 datacube by fetching live satellite data."""
    print(f"Starting LIVE Datacube Generation for prediction at ({lat}, {lon})")

    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    num_days = DATACUBE_CONFIG['temporal_extent_days']
    datacube_3d = np.zeros((grid_size, grid_size, num_days))

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format. Please use YYYY-MM-DD.")
        return None

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

        daily_data_points = []
        # Process all downloaded files for the day and merge them
        for file_path in files:
            extracted_data = extract_modality_from_granule(file_path, spatial_bounds)
            if extracted_data:
                daily_data_points.append(extracted_data)
            
            try:
                Path(file_path).unlink()
                print(f"Deleted cache file: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

        if not daily_data_points:
            print(f"Found granules, but no valid chlor_a data within bounds.")
            continue

        # Combine data from all granules for the day
        all_lats = np.concatenate([d['lats'] for d in daily_data_points])
        all_lons = np.concatenate([d['lons'] for d in daily_data_points])
        all_values = np.concatenate([d['values'] for d in daily_data_points])

        # Reproject the combined points onto 15x15 grid
        image_2d = reproject_to_grid(all_lats, all_lons, all_values, spatial_bounds)
        datacube_3d[:, :, day_offset] = image_2d
        print(f"Success! Populated datacube for this day.")

    print(f"Datacube Generation Complete. Final shape: {datacube_3d.shape}")
    return datacube_3d