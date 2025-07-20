import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import netCDF4 as nc
from scipy.interpolate import griddata
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pyproj import Transformer

nc_lock = threading.Lock()
# Define datacube config (must match with trained model settings)
DATACUBE_CONFIG = {
    'spatial_extent_km': 100,
    'temporal_extent_days': 10,
    'spatial_resolution_km': 2,
}
HABNET_MODIS_AQUA_MODALITIES = [
    'chlor_a',      
    'Rrs_412',     
    'Rrs_443'      
]

def get_utm_zone_from_coords(lat, lon):
    """Determines the correct UTM zone EPSG code from lat/lon."""
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = f"326{utm_zone:02d}" if lat >= 0 else f"327{utm_zone:02d}"
    return epsg_code

def setup_utm_projection(lat, lon):
    """Sets up the forward transformer for a location's UTM zone."""
    if not Transformer: 
        return None, None
    epsg_code = get_utm_zone_from_coords(lat, lon)
    transformer_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    return transformer_to_utm, epsg_code

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

def extract_modality_from_granule(file_path, modality, spatial_bounds, transformer_to_utm):
    """Extracts a single modality's data and projects its coordinates to UTM."""
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
                
                final_lons = lons[spatial_mask][valid_mask]
                final_lats = lats[spatial_mask][valid_mask]
                final_vals = vals_roi[valid_mask]
                
                utm_x, utm_y = transformer_to_utm.transform(final_lons, final_lats)
                return {'utm_x': utm_x, 'utm_y': utm_y, 'values': final_vals}
    except Exception as e:
        print(f"Error reading modality {modality} from {Path(file_path).name}: {e}")
    return None

def reproject_to_grid(utm_x, utm_y, values, center_utm_x, center_utm_y):
    """Reprojects scattered UTM data points onto a regular 50x50 grid."""
    grid_size = int(DATACUBE_CONFIG['spatial_extent_km'] / DATACUBE_CONFIG['spatial_resolution_km'])
    half_extent_m = DATACUBE_CONFIG['spatial_extent_km'] * 1000 / 2
    resolution_m = DATACUBE_CONFIG['spatial_resolution_km'] * 1000

    x_coords = np.linspace(center_utm_x - half_extent_m + resolution_m / 2, center_utm_x + half_extent_m - resolution_m / 2, grid_size)
    y_coords = np.linspace(center_utm_y - half_extent_m + resolution_m / 2, center_utm_y + half_extent_m - resolution_m / 2, grid_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    source_points = np.column_stack((utm_x, utm_y))
    target_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
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

def process_single_day(day_offset, start_date, spatial_bounds, center_utm_x, center_utm_y, transformer_to_utm):
    """
    Encapsulates all the work needed for one day.
    It will be executed in a separate thread for each of the 10 days.
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
    files = earthaccess.download([granules[0]], local_path=str(raw_dir))

    for modality in HABNET_MODIS_AQUA_MODALITIES:
        modality_points = [d for f in files if (d := extract_modality_from_granule(f, modality, spatial_bounds, transformer_to_utm)) is not None]
    
        if not modality_points:
            continue

        all_utm_x = np.concatenate([d['utm_x'] for d in modality_points])
        all_utm_y = np.concatenate([d['utm_y'] for d in modality_points])
        all_values = np.concatenate([d['values'] for d in modality_points])

        image_2d = reproject_to_grid(all_utm_x, all_utm_y, all_values, center_utm_x, center_utm_y)
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
    """Generates a single 50x50x10x7 datacube by fetching live satellite data."""
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

    # Setup UTM projection based on the requested location
    transformer_to_utm, _ = setup_utm_projection(lat, lon)
    if not transformer_to_utm:
        return None
    center_utm_x, center_utm_y = transformer_to_utm.transform(lon, lat)

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
        future_to_day = {
            executor.submit(process_single_day, day, start_date, spatial_bounds, center_utm_x, center_utm_y, transformer_to_utm): day for day in range(num_days)}

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