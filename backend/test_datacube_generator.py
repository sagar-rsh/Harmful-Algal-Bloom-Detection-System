from datacube_generator import generate_prediction_datacube
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

datacube = generate_prediction_datacube(lat, lon, start_date)
print("--- %s seconds ---" % (time.time() - start_time))
print(f'\n{datacube.shape}\n')
# print(f'\n{datacube}')