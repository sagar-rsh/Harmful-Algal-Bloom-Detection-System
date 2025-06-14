from flask import Flask, request, jsonify
from keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Model Loading
# Load the trained model
MODEL_PATH = 'models/hab_model.h5' # local path for now. Final version will fetch the model from S3.
try:
	model = load_model(MODEL_PATH)
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
	print(data)

	return jsonify({"success": "Request received."}), 200


if __name__ == '__main__':
	# Run the app on port 5000
	app.run(host='127.0.0.1', port=5000, debug=True)
