import requests
import pandas as pd
import sklearn
import json
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, RocCurveDisplay
import joblib

# Step 1: Fetch the Data
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 38.90, # Washington, DC
    "longitude": -77.04,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Shape", df.shape)
print(df.head())
print(df.tail())

# Step 2: Engineer Labels
# Define "good for running" conditions:
# temperature_2m_max: 7 - 26 °C (45-79°F)
# temperature_2m_min: ≥ 0 °C (above freezing)
# precipitation_sum: < 3.0 mm
# wind_speed_10m_max: < 30 km/h

df["good_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 26) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3.0) &
    (df["wind_speed_10m_max"] < 30)   
    
)
print(df.head())
# Convert True/False labels into 1/0
df["good_running"] = df["good_running"].astype(int)
print(df.head())

# Check class distribution
print(df["good_running"].value_counts())
# Check percentage of good running days
print(df["good_running"].mean())

# In Washington, DC, about 40% of days were labeled as good for running,
# and this seems reasonable because DC has cold winters, hot summers, and rainy days.

# Step 3: Train and Tune
# Separate Features 'X' and Labels 'y'
# Features
X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
]

# Label (what we want to predict)
y = df["good_running"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Create pipeline 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"

)

grid_search.fit(X_train, y_train)

# Print the best model
print("Best C:", grid_search.best_params_["model__C"])
print("Best CV AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_

# Test set evaluation
y_pred = best_model.predict(X_test)
y_probs = best_model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(
    y_test,
    y_probs
)

print("Test AUC:", test_auc)
print(classification_report(y_test, y_pred))

# Create ROC curve for weather classifier
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probs
)

plt.figure(figsize=(6,5))

RocCurveDisplay.from_predictions(
    y_test,
    y_probs,
    name="Logistic Regression"
    )
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")

plt.title("Weather Classifier ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("assignments_04/outputs/weather_roc.png")
plt.close()

# Step 4: Reflect on Evaluation

# The AUC score shows how well the model separates good running days from bad running days.
# The model performance is acceptable, but it is not perfect because the test AUC is around 0.70.
# The model has more false positives than false negatives for good running days.
# I would not always use the default threshold of 0.5 because the best threshold depends on the goal of the application.

# Step 5: Save the Model
# Save the model 
joblib.dump(best_model, "assignments_04/models/weather_classifier.pkl")

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ],

    "label": "good_running",
    "best_params": grid_search.best_params_,
    "test_auc": round(test_auc, 4),
    "trained_on": "2023 Open-Meteo data",
    "city": {
        "name": "Washington, DC",
        "latitude": 38.90,
        "longitude": -77.04 
    },
    
    "label_thresholds": {
        "temperature_2m_max": "7-26°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
    },
}

with open("assignments_04/models/weather_classifier_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model and Metadata saved sucessfully.")
