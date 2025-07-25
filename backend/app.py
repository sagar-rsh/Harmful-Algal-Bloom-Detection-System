from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
# from keras.models import load_model
import numpy as np
from datetime import date
import earthaccess
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from datacube_generator import generate_prediction_datacube
from image_processor import convert_datacube_to_images
from joblib import load
from mangum import Mangum
from dependencies import get_api_key
from enum import Enum
from config import TIER_CONFIG

# Initialize FastAPI
app = FastAPI(title="HAB Prediction API")

models = {}
for tier, config in TIER_CONFIG.items():
	try:
		path = config["model_path"]
		if path.endswith(".pkl"):
			models[tier] = load(path)
		# elif path.endswith(".h5"):
		# 	models[tier] = load_model(path)
		print(f"Loaded model for {tier} from {path}")
	except Exception as e:
		print(f"ERROR loading model for {tier}: {e}")
		models[tier] = None

# Load environment variables
load_dotenv()
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

try:
	earthaccess.login(strategy="environment")
except Exception as e:
	print(f"Error: NASA Earthdata login failed: {e}")

label_map = {0: "non-toxic", 1: "toxic"}

# This provides automatic data validation and documentation.
class Tier(str, Enum):
	FREE = "free"
	TIER_1 = "tier1"
	TIER_2 = "tier2"

class PredictionRequest(BaseModel):
	latitude: float
	longitude: float
	date: str
	tier: Tier

class ConfidenceScores(BaseModel):
	non_toxic: str
	toxic: str

class PredictionResponse(BaseModel):
	prediction_for_date: str
	predicted_label: str
	confidence_scores: ConfidenceScores

# API endpoint definition
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, api_key: str = Depends(get_api_key)):
	print(f"Received prediction request for tier: {request.tier}")

	# Select the config for the requested tier
	config = TIER_CONFIG[request.tier.value]
	# Select the pre-loaded model for the tier
	model = models[request.tier.value]
	if model is None:
		raise HTTPException(status_code=500, detail=f"Model for tier '{request.tier.value}' is not loaded.")
	
	try:
		lat = request.latitude
		lon = request.longitude
		start_date_str = request.date

		# Generate the datacube with the specific config
		datacube = generate_prediction_datacube(lat, lon, start_date_str, config)

		if datacube is None:
			raise HTTPException(status_code=400, detail="Failed to generate datacube")
		
		# Convert to resized, normalized, concatenated image sequence
		# The output shape is now (10, 64, 64, 9)
		image_sequence = convert_datacube_to_images(datacube, config)

		'''
		# Prepare sequence for model
		# Current shape is 4D: (time_steps, height, width, channels)-> (10, 64, 64, 9), we need to add the batch dimension: (batch_size, time_steps, height, width, channels) for the model -> (1, 10, 64, 64, 9)
		# batch: How many datacubes we are predicting at once
		'''
		# # Adding batch dimension - (1, 10, 64, 64, 9)
		# model_input = np.expand_dims(image_sequence, axis=0)
			
		# # The training code scales by 255.0, so we do the same here
		# model_input = model_input.astype(np.float32) / 255.0
		        # Flatten the (10, 64, 64, 9) array into a 1D vector
		flattened_features = image_sequence.flatten()
		
		# Reshape it into a 2D array with one row for the model
		model_input = flattened_features.reshape(1, -1)

		print(f"Final model input shape: {model_input.shape}, dtype: {model_input.dtype}")

		# Model will return a list of probabilities ([0.42, 0.58])
		# prediction_probs = model.predict(model_input)[0]
		prediction_probs = model.predict_proba(model_input)[0]
		print(prediction_probs)
		# Extract the probabilities for non-toxic and toxic
		prob_non_toxic = prediction_probs[0]
		prob_toxic = prediction_probs[1]

		# Determine the predicted label
		predicted_class_index = np.argmax(prediction_probs)
		predicted_label = label_map[predicted_class_index]

		# Calculate the date that was predicted for (start_date + 5 days)
		prediction_target_date = datetime.strptime(start_date_str, '%Y-%m-%d') + timedelta(days=10)

		# Return a Pydantic model, which FastAPI automatically converts to JSON.
		return PredictionResponse(
			prediction_for_date=prediction_target_date.strftime('%Y-%m-%d'),
			predicted_label=predicted_label,
			confidence_scores=ConfidenceScores(
					non_toxic=f"{prob_non_toxic:.4f}",
					toxic=f"{prob_toxic:.4f}"
			)
		)

	except Exception as e:
		print(f"ERROR during prediction: {e}")
		raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
	
handler = Mangum(app, lifespan="off")
