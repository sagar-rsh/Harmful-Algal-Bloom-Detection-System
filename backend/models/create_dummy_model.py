import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


def create_and_save_model():
	"""
	Creates a dummy classifier model based on metadata.csv data structure,
	trains it on sample data, and saves it to a file.
	"""
	# Sample data mimicking the structure of metadata.csv
	data = {
		'region': ['midwest', 'west', 'south', 'south', 'midwest', 'midwest', 'south', 'south', 'south', 'south'],
		'latitude': [39.08031907, 36.5597, 35.87508324, 35.487, 38.04947082, 39.474744, 35.64774247, 35.90688526, 35.72652194, 35.98],
		'longitude': [-86.43086667, -121.51, -78.87843428, -79.06213296, -99.82700111, -86.898353, -79.2717821, -79.13296163, -79.12545778, -78.79168619],
		'distance_to_water_m': [0.0, 3512.0, 514.0, 129.0, 19.0, 0.0, 743.0, 774.0, 697.0, 377.0],
		'severity': [0, 1, 0, 0, 1, 1, 0, 0, 0, 1]  # Target variable
	}
	df = pd.DataFrame(data)
	X = df[['region', 'latitude', 'longitude', 'distance_to_water_m']]
	y = df['severity']

	# Preprocessing: encode categorical and scale numeric
	preprocessor = ColumnTransformer([
		('cat', OneHotEncoder(handle_unknown='ignore'), ['region']),
		('num', StandardScaler(), ['latitude', 'longitude', 'distance_to_water_m'])
	])

	pipeline = Pipeline([
		('preprocessor', preprocessor),
		('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
	])

	pipeline.fit(X, y)
	print("Dummy model trained successfully.")

	# Save the model
	save_dir = './'
	model_path = os.path.join(save_dir, 'dummy_hab_model.joblib')
	joblib.dump(pipeline, model_path)

	print(f"Model saved to {model_path}")

if __name__ == '__main__':
	create_and_save_model()