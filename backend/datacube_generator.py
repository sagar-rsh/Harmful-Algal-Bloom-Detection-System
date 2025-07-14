import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import netCDF4 as nc
from scipy.interpolate import griddata
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

nc_lock = threading.Lock()
# Define datacube config (must match with trained model settings)
DATACUBE_CONFIG = {
    'spatial_extent_km': 30,
    'temporal_extent_days': 5,
    'spatial_resolution_km': 2,
}
HABNET_MODIS_AQUA_MODALITIES = [
    'chlor_a',      
    'Rrs_412',     
    'Rrs_443',      
    'Rrs_488',      
    'Rrs_531',    
    'Rrs_555',      
    'par'           
]

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

def extract_modality_from_granule(file_path, modality, spatial_bounds):
    """Extracts chlorophyll-a data from a single downloaded .nc file."""
    try:
        with nc_lock:
            with nc.Dataset(file_path, 'r') as ds:
                if 'geophysical_data' not in ds.groups or 'navigation_data' not in ds.groups or modality not in ds.groups['geophysical_data'].variables:
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
                
                vals_roi = geo_data.variables[modality][:][spatial_mask]
                if hasattr(vals_roi, 'mask'):
                    valid_mask = ~vals_roi.mask
                else:
                    valid_mask = np.isfinite(vals_roi)
                
                if modality == 'chlor_a':
                    valid_mask = valid_mask & (vals_roi > 0) & (vals_roi < 1000)
                elif modality.startswith('Rrs_'):
                    valid_mask = valid_mask & (vals_roi > 0) & (vals_roi < 1.0)
                elif modality == 'par':
                    valid_mask = valid_mask & (vals_roi > 0)

                if not np.any(valid_mask):
                    return None
                
                return {'lats': lats[spatial_mask][valid_mask], 'lons': lons[spatial_mask][valid_mask], 'values': vals_roi[valid_mask]}
    except Exception as e:
        print(f"Error reading modality {modality} from {Path(file_path).name}: {e}")
    return None

def reproject_to_grid(lats, lons, values, spatial_bounds):
    """Reprojects scattered satellite data points onto a regular grid."""
    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    target_lats = np.linspace(spatial_bounds['lat_min'], spatial_bounds['lat_max'], grid_size)
    target_lons = np.linspace(spatial_bounds['lon_min'], spatial_bounds['lon_max'], grid_size)
    grid_lons, grid_lats = np.meshgrid(target_lons, target_lats)

    source_points = np.column_stack((lons.ravel(), lats.ravel()))
    target_points = np.column_stack((grid_lons.ravel(), grid_lats.ravel()))
    
    if len(values) < 4: 
        return np.zeros((grid_size, grid_size))

    # interpolate to grid with linear
    interpolated = griddata(
        source_points, values,
        target_points, method='linear', fill_value=np.nan
    )
    # fill leftover NaNs with nearest neighbor if possible
    if np.any(np.isnan(interpolated)):
        interpolated_nn = griddata(source_points, values, target_points, method='nearest', fill_value=0.0)
        nan_mask = np.isnan(interpolated)
        interpolated[nan_mask] = interpolated_nn[nan_mask]

    return interpolated.reshape((grid_size, grid_size))

def process_single_day(day_offset, start_date, spatial_bounds):
    """
    Encapsulates all the work needed for one day.
    It will be executed in a separate thread for each of the 5 days.
    """
    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    target_date = start_date + timedelta(days=day_offset)
    print(f"[Thread] Starting work for Day {day_offset + 1}: {target_date.strftime('%Y-%m-%d')}")
    daily_images = {mod: np.zeros((grid_size, grid_size)) for mod in HABNET_MODIS_AQUA_MODALITIES}

    granules = search_modis_l2_data(target_date, spatial_bounds)
    if not granules:
        print(f"[Thread] No satellite data found for Day {day_offset + 1}.")
        return day_offset, daily_images # Return an empty grid
    
    # Download the found granules to a local cache
    raw_dir = Path("habnet_data_cache")
    raw_dir.mkdir(exist_ok=True)
    files = earthaccess.download(granules, local_path=str(raw_dir))

    for modality in HABNET_MODIS_AQUA_MODALITIES:
        modality_points = [d for f in files if (d := extract_modality_from_granule(f, modality, spatial_bounds)) is not None]
    
        if not modality_points:
            continue

        all_lats = np.concatenate([d['lats'] for d in modality_points])
        all_lons = np.concatenate([d['lons'] for d in modality_points])
        all_values = np.concatenate([d['values'] for d in modality_points])

        image_2d = reproject_to_grid(all_lats, all_lons, all_values, spatial_bounds)
        daily_images[modality] = image_2d

    print(f"[Thread] Finished work for Day {day_offset + 1}.")
    # Clean up
    for file_path in files:
        try:
            Path(file_path).unlink()
            print(f"Deleted cache file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")
    
    # Return the day's index and the processed 2D image
    return day_offset, daily_images

def generate_prediction_datacube(lat, lon, start_date_str):
    """Generates a single 15x15x7 datacube by fetching live satellite data."""
    print(f"Starting Concurrent Datacube Generation for prediction at ({lat}, {lon})")

    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    num_days = DATACUBE_CONFIG['temporal_extent_days']
    num_modalities = len(HABNET_MODIS_AQUA_MODALITIES)

    final_datacube = np.zeros((grid_size, grid_size, num_days, num_modalities))

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

    # Run tasks for each day in parallel
    # The number of workers (threads) is set to num_days (one for each day) for now (might need to update it later)
    with ThreadPoolExecutor(max_workers=num_days) as executor:
        # This creates a list of future objects, which are placeholders for the results.
        future_to_day = {executor.submit(process_single_day, day, start_date, spatial_bounds): day for day in range(num_days)}

        # as_completed waits for any of the futures to finish.
        for future in as_completed(future_to_day):
            try:
                # Get the results
                day_offset, daily_images = future.result() # tuple (day_offset, daily_images)
                for mod_idx, modality in enumerate(HABNET_MODIS_AQUA_MODALITIES):
                    final_datacube[:, :, day_offset, mod_idx] = daily_images[modality]
            except Exception as e:
                print(f'Day {future_to_day[future] + 1} generated an exception: {e}')

    print(f"Datacube Generation Complete. Final shape: {final_datacube.shape}")
    return final_datacube