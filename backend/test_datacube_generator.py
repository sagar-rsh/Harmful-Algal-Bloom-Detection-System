from datacube_generator import generate_prediction_datacube
from image_processor import convert_datacube_to_images
import numpy as np
import earthaccess
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys
import time

# Load environment variables
load_dotenv()
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

try:
	earthaccess.login(strategy="environment")
except Exception as e:
	print(f"Error: NASA Earthdata login failed: {e}")
	sys.exit(0)

start_time = time.time()
lat, lon = 27.77624, -82.77021
start_date = "2015-11-09"
config = {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 6,
        "model_path": "models/Tier-1_model.h5"
    }

datacube = generate_prediction_datacube(lat, lon, start_date, config)
print(f'\n{datacube.shape}\n')
# print(f'\n{datacube}')

image_sequence = convert_datacube_to_images(datacube, config)
print(f'\n{image_sequence.shape}\n')

print("--- %s seconds ---" % (time.time() - start_time))