import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

# === Configs ===
test_path = "data/processed/test_clean.csv"
model_dir = "models"
output_dir = "charts/accuracy"

# === Ensure directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load test data ===
test_df = pd.read_csv(test_path)

feature_cols = [
    'year', 'population', 'gdp',
    'cement_co2_per_capita', 'co2_growth_abs', 'co2_including_luc_growth_abs',
    'co2_including_luc_per_gdp', 'co2_including_luc_per_unit_energy',
    'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2_per_capita',
    'energy_per_capita', 'flaring_co2_per_capita',
    'nitrous_oxide_per_capita', 'temperature_change_from_n2o'
]

log_transform_cols = ['population', 'gdp']
for col in log_transform_cols:
    test_df[col] = np.log1p(test_df[col])

X_test = test_df[feature_cols]
y_test = test_df["co2"]
years = test_df["year"]

# === Generate accuracy plots per model ===
for fname in os.listdir(model_dir):
    if fname.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, fname)
        model_name = fname.replace("_best_pipeline.joblib", "")
        pipeline = joblib.load(model_path)

        y_pred = pipeline.predict(X_test)

        # === Save line chart ===
        plt.figure(figsize=(8, 5))
        plt.plot(years, y_test, label="Actual", marker='o', color="#3b82f6")
        plt.plot(years, y_pred, label="Predicted", marker='o', linestyle="--", color="#10b981")
        plt.title(f"{model_name.capitalize()} Prediction Accuracy")
        plt.xlabel("Year")
        plt.ylabel("CO₂ Emissions (MT)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"{model_name}_accuracy.png")
        plt.savefig(chart_path)
        plt.close()

print("Accuracy charts saved to:", output_dir)
