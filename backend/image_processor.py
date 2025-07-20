import numpy as np
import cv2

# Current model expects 64x64 (HxW)
MODEL_IMG_SIZE = (64, 64)
# Using only 3 modalities for now (Current model is trained on 3 modalities)
HABNET_MODIS_AQUA_MODALITIES = [
    'chlor_a', 'Rrs_412', 'Rrs_443'
]


def calculate_min_max_from_datacube(numerical_datacube):
    """
    Calculates the min and max value for each modality channel within a single datacube.
    """
    modality_min_max = {}
    num_modalities = numerical_datacube.shape[3]

    for mod_idx in range(num_modalities):
        modality_name = HABNET_MODIS_AQUA_MODALITIES[mod_idx]
        modality_data = numerical_datacube[:, :, :, mod_idx]
        
        if np.all(np.isnan(modality_data)) or not np.any(np.isfinite(modality_data)):
             min_val, max_val = 0, 1
        else:
            min_val = np.nanmin(modality_data[np.isfinite(modality_data)])
            max_val = np.nanmax(modality_data[np.isfinite(modality_data)])
        
        modality_min_max[modality_name] = (min_val, max_val)
        
    return modality_min_max


def normalize_slice(image_slice, min_val, max_val):
    """
    Normalizes a single 50x50 slice of data to an 8-bit BGR image.
    """
    if abs(max_val - min_val) < 1e-10:
        gray_slice = np.full(image_slice.shape, 128, dtype=np.uint8)
    else:
        normalized = (image_slice - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        normalized = np.nan_to_num(normalized, nan=0.0)
        gray_slice = (normalized * 255).astype(np.uint8)
    
    # Convert grayscale to 3 channel BGR
    bgr_slice = cv2.cvtColor(gray_slice, cv2.COLOR_GRAY2BGR)
    return bgr_slice


def convert_datacube_to_images(numerical_datacube):
    """
    Converts a raw numerical datacube into a final sequence of concatenated images.
    The final shape will be (10, 64, 64, 9)
    """
    print("Converting numerical datacube to image format...")
    
    local_min_max = calculate_min_max_from_datacube(numerical_datacube)
    print(f"Calculated local min/max ranges: {local_min_max}")

    height, width, num_days, num_modalities = numerical_datacube.shape
    
    # This will hold the final sequence of 10 images
    final_sequence = []

    for day_idx in range(num_days):
        daily_modality_images = []
        for mod_idx in range(num_modalities):
            modality_name = HABNET_MODIS_AQUA_MODALITIES[mod_idx]
            min_val, max_val = local_min_max[modality_name]
            
            numerical_slice = numerical_datacube[:, :, day_idx, mod_idx]
            
            # Normalize to a (50, 50, 3) BGR image
            bgr_slice = normalize_slice(numerical_slice, min_val, max_val)
            
            # Resize to a (64, 64, 3) BGR image
            resized_bgr_slice = cv2.resize(bgr_slice, MODEL_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            
            daily_modality_images.append(resized_bgr_slice)
        
        # Concatenate the modality images for this day
        # This combines three (64, 64, 3) images into one (64, 64, 9) image
        concatenated_day_image = np.concatenate(daily_modality_images, axis=-1)
        final_sequence.append(concatenated_day_image)
        
    # Stack the 10 daily images to create the final sequence
    final_sequence_array = np.stack(final_sequence, axis=0)
    
    print(f"Image conversion and resizing complete. Final shape: {final_sequence_array.shape}")
    return final_sequence_array