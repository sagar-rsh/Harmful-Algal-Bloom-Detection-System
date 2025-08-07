import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time

# === Configurable Parameters ===
CSV_PATH = "sample_data.csv"
API_URL = "http://127.0.0.1:5000/predict"
START_INDEX = 1300
END_INDEX = 1350 
DELAY_BETWEEN_CALLS = 1  # seconds
RESULT_CSV_PATH = Path(f"prediction_evaluation_results_{START_INDEX}_{END_INDEX}.csv")

label_map = {"non-toxic": 0, "toxic": 1}
reverse_label_map = {0: "non-toxic", 1: "toxic"}

# Load and Preprocess Data
df = pd.read_csv(CSV_PATH)
df['SAMPLE_DATE'] = pd.to_datetime(df['SAMPLE_DATE'], format="%d/%m/%Y")
subset_df = df.iloc[START_INDEX:END_INDEX]

results = []

correct = 0
total = 0
failed_requests = 0

print(f"Evaluating rows {START_INDEX} to {END_INDEX or len(df)}...")
print("=" * 60)

for idx, row in subset_df.iterrows():
    latitude = row['LATITUDE']
    longitude = row['LONGITUDE']
    target_date = row['SAMPLE_DATE']
    hab_event = row['HAB_EVENT']

    start_date = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")

    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "date": start_date
    }

    result_record = {
        "row_index": idx,
        "input_lat": latitude,
        "input_lon": longitude,
        "target_date": target_date.strftime("%Y-%m-%d"),
        "start_date": start_date,
        "ground_truth": hab_event
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()

        predicted_label = result['predicted_label']
        predicted_value = label_map.get(predicted_label, -1)
        prob_non_toxic = float(result['confidence_scores']['non_toxic'])
        prob_toxic = float(result['confidence_scores']['toxic'])

        is_correct = int(predicted_value == hab_event)

        result_record.update({
            "predicted_label": predicted_label,
            "predicted_value": predicted_value,
            "prob_non_toxic": prob_non_toxic,
            "prob_toxic": prob_toxic,
            "is_correct": is_correct
        })

        print(f"[{idx}] GT: {hab_event}, Pred: {predicted_value} ({predicted_label}) --> {'✓' if is_correct else '✗'}")
        correct += is_correct

    except Exception as e:
        print(f"[{idx}] ❌ Request failed: {e}")
        result_record.update({
            "predicted_label": "error",
            "predicted_value": -1,
            "prob_non_toxic": None,
            "prob_toxic": None,
            "is_correct": 0
        })
        failed_requests += 1

    total += 1
    results.append(result_record)
    time.sleep(DELAY_BETWEEN_CALLS)

# Save Results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_CSV_PATH, index=False)
print(f"\nResults saved to: {RESULT_CSV_PATH.absolute()}")

# Summary
print("=" * 60)
print(f"Completed evaluation")
print(f"Total processed:         {total}")
print(f"Successful predictions:  {correct}")
print(f"Failed requests:         {failed_requests}")
accuracy = (correct / total) * 100 if total > 0 else 0.0
print(f"Prediction accuracy:     {accuracy:.2f}%")