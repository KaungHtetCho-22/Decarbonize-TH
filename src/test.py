import pandas as pd
import numpy as np
import joblib
import os

# === Paths ===
model_dir = "models"

# === Define log-transform columns only (the pipeline handles the rest)
log_transform_cols = ['population', 'gdp']

# === Sample input (2023 data) ===
data_2023 = {
    'year': [2023],
    'population': [71702438.0],
    'gdp': [1124143726592.0],
    'cement_co2_per_capita': [0.293],
    'co2_growth_abs': [5.432],
    'co2_including_luc_growth_abs': [4.502],
    'co2_including_luc_per_gdp': [0.273],
    'co2_including_luc_per_unit_energy': [0.221],
    'co2_per_gdp': [0.242],
    'co2_per_unit_energy': [0.196],
    'coal_co2_per_capita': [0.973],
    'energy_per_capita': [19357.754],
    'flaring_co2_per_capita': [0.006],
    'nitrous_oxide_per_capita': [0.339],
    'temperature_change_from_n2o': [0.001],
    'country': ['Thailand'],
    'co2': [264.389]
}

df = pd.DataFrame(data_2023)

# === Apply log1p transform ONLY to raw features
for col in log_transform_cols:
    df[col] = np.log1p(df[col])

# === Drop unused columns
df_input = df.drop(columns=["country", "co2"])

# === Run predictions using full pipeline ===
results = []

for filename in os.listdir(model_dir):
    if filename.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, filename)
        pipeline = joblib.load(model_path)  # includes scaler + model

        y_pred = pipeline.predict(df_input)

        results.append({
            "model": filename.replace("_best_pipeline.joblib", ""),
            "predicted_co2": round(y_pred[0], 2)
        })

# === Display results ===
results_df = pd.DataFrame(results).sort_values("predicted_co2")
print("\n=== Model Comparison: Predicted CO₂ for Thailand in 2023 ===")
print('\nActual CO₂ = 264.389 ===')
print(results_df.to_string(index=False))
