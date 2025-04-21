import pandas as pd
import numpy as np
import joblib
import os

# === Paths ===
model_dir = "models"
target_scaler_path = "artifacts/target_scaler.pkl"

# === Load scaler ===
target_scaler = joblib.load(target_scaler_path)

# === Define columns ===
log_transform_cols = [
    'population', 'gdp', 'primary_energy_consumption',
    'oil_co2', 'coal_co2', 'cement_co2',
    'total_ghg', 'co2_including_luc'
]
feature_cols = log_transform_cols + ["temperature_change_from_ghg"]

# === Sample input (2023 data) ===
data_2023 = {
    'year': [2023],
    'population': [71702438.0],
    'gdp': [1124143726592.0],
    'primary_energy_consumption': [1390.812],
    'oil_co2': [150.343],
    'coal_co2': [59.327],
    'cement_co2': [19.248],
    'total_ghg': [416.852],
    'co2_including_luc': [297.369],
    'temperature_change_from_ghg': [0.015],
    'co2': [264.389], 
    'country': ['Thailand']
}
df = pd.DataFrame(data_2023)

# === Apply log1p transform ===
for col in log_transform_cols:
    df[col] = np.log1p(df[col])
X_new = df[feature_cols]

# === Collect predictions ===
results = []

for filename in os.listdir(model_dir):
    if filename.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, filename)
        model = joblib.load(model_path)
        
        # Predict
        y_scaled = model.predict(X_new)
        y_log = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
        y_pred = y_log  # No np.expm1 needed since target was not log-transformed

        results.append({
            "model": filename.replace("_best_pipeline.joblib", ""),
            "predicted_co2": round(y_pred[0], 2)
        })

# === Display results ===
results_df = pd.DataFrame(results).sort_values("predicted_co2")
print("\n=== Model Comparison: Predicted CO₂ for Thailand in 2023 ===")
print('\nActual results = 264.389 ===')
print(results_df.to_string(index=False))
