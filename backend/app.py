from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import earthaccess
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from datacube_generator import generate_prediction_datacube

# Initialize the Flask application
app = Flask(__name__)

# Model Loading:
# Load the trained model
MODEL_PATH = 'models/hab_model.keras' # local path for now. Final version will fetch the model from S3.
try:
	model = load_model(MODEL_PATH)
	print("Model loaded successfully.")
except FileNotFoundError:
	print(f"Error: Model file not found at {MODEL_PATH}.")
	model = None

# Load environment variables
load_dotenv()
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

try:
	earthaccess.login(strategy="environment")
except Exception as e:
	print(f"Error: NASA Earthdata login failed: {e}")

label_map = {0: "non-toxic", 1: "toxic"}

# API endpoint definition
@app.route('/predict', methods=['POST'])
def predict():
	if model is None:
		return jsonify({"error": "Model is not loaded. Please check server logs."}), 500
	
	if not request.json:
		return jsonify({"error": "Invalid input: No JSON body received."}), 400
	
	# Extract data from the request
	data = request.json

	required_keys = ['latitude', 'longitude', 'date']
	if not all(key in data for key in required_keys):
		return jsonify({"error": f"Missing one of the required keys: {required_keys}"}), 400

	lat = data['latitude']
	lon = data['longitude']
	start_date_str = data['date']

	try:
		# Generate 15x15x5 datacube
		datacube = generate_prediction_datacube(lat, lon, start_date_str)

		if datacube is None:
			return jsonify({"error": "Failed to generate datacube. Check input date format."}), 400
		
		'''
		# Prepare datacube for model
		# Current datacube shape is 3D: (width, height, time_steps)-> (15, 15, 5), we need 4D input: (batch_size, width, height, time_steps) for the model
		# batch: How many datacubes we are predicting at once
		'''
		# Adding batch dimension - (1, 15, 15, 5)
		model_input = np.expand_dims(datacube, axis=0)

		print(f"Final model input shape: {model_input.shape}")

		# Model will return a list of probabilities ([0.42, 0.58])
		prediction_probs = model.predict(model_input)[0]
		# Extract the probabilities for non-toxic and toxic
		prob_non_toxic = prediction_probs[0]
		prob_toxic = prediction_probs[1]

		# Determine the predicted label
		predicted_class_index = np.argmax(prediction_probs)
		predicted_label = label_map[predicted_class_index]

		# Calculate the date that was predicted for (start_date + 5 days)
		prediction_target_date = datetime.strptime(start_date_str, '%Y-%m-%d') + timedelta(days=5)

		# Format the response
		result = {
			"prediction_for_date": prediction_target_date.strftime('%Y-%m-%d'),
			"predicted_label": predicted_label,
			"confidence_scores": {
				"non_toxic": f"{prob_non_toxic:.4f}",
				"toxic": f"{prob_toxic:.4f}"
			}
		}

		return jsonify(result), 200

	except Exception as e:
		return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500
	

if __name__ == '__main__':
	# Run the app on port 5000
	app.run(host='0.0.0.0', port=5000)
