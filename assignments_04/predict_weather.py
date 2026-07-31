import joblib
import json
import pandas as pd
# Task 1: Load and Verify
# Load the pipeline

model = joblib.load("assignments_04/models/weather_classifier.pkl")
with open("assignments_04/models/weather_classifier_metadata.json", "r") as f:
    metadata = json.load(f)

print("City:", metadata["city"]["name"])
print("Features:", metadata["features"])
print("Test AUC:", metadata["test_auc"])

# Task 2: Predict on New Data
new_days = pd.DataFrame({
    "temperature_2m_max": [18.0, 2.0, 28.0, 24.0, 7.0],
    "temperature_2m_min": [10.0, -2.0, 19.0, 12.0, 0.0],
    "precipitation_sum":  [0.0, 0.0, 12.0, 1.0, 2.9],
    "wind_speed_10m_max": [18.0, 8.0, 35.0, 25.0, 29.0]
})

predictions = model.predict(new_days)
probabilities = model.predict_proba(new_days)[:, 1]

for i in range(len(new_days)):
    print(f" Day {i+1}")
    print(new_days.iloc[i])
    print("Predicted:", "Good" if predictions[i] == 1 else "Skip")
    print(f"Confidence: {probabilities[i]:.2f}")

# Task 3: Reflect
# 1. The borderline case had a probability of about 0.55. 
# This is close to 0.50, so the model is not very confident. 
# 2. The model and metadata files would not exist,
# causing a file not found error. 
# 3. It would fetch tomorrow's forecast from the Open-Meteo Forecast API 
# instead of using hard-coded hypothetical days.
