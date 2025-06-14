from flask import Flask, request, jsonify
from keras.models import load_model
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Model Loading:
# Load the trained model
MODEL_PATH = 'models/dummy_hab_model.joblib' # local path for now. Final version will fetch the model from S3.
try:
	# model = load_model(MODEL_PATH)
	model = joblib.load(MODEL_PATH) # Remove this for tensorflow model
	print("Model loaded successfully.")
except FileNotFoundError:
	print(f"Error: Model file not found at {MODEL_PATH}.")
	model = None

# API endpoint definition
@app.route('/predict', methods=['POST'])
def predict():
	if model is None:
		return jsonify({"error": "Model is not loaded. Please check server logs."}), 500
	
	if not request.json:
		return jsonify({"error": "Invalid input: No JSON body received."}), 400
	
	# Extract data from the request
	data = request.json
	# print(data)

	required_keys = ['region', 'latitude', 'longitude', 'distance_to_water_m']
	if not all(key in data for key in required_keys):
		return jsonify({"error": f"Missing one of the required keys: {required_keys}"}), 400
	
	try:
		input_df = pd.DataFrame([data])
		input_df = input_df[required_keys]

		prediction_value = int(model.predict(input_df)[0])
		prediction_proba = model.predict_proba(input_df)[0]

		# Format the response
		result = {
			"predicted_severity": prediction_value,
            "confidence_scores": {
                str(i): round(float(prob), 4) for i, prob in enumerate(prediction_proba)
            }
		}

		return jsonify(result), 200

	except Exception as e:
		return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500
	

if __name__ == '__main__':
	# Run the app on port 5000
	app.run(host='127.0.0.1', port=5000, debug=True)
