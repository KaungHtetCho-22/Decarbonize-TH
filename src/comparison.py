import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import json

# === Configs ===
test_path = "data/processed/test_clean.csv"
model_dir = "models"
output_dir = "charts/accuracy"
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

# Filter for years >= 2000
filtered_df = test_df[test_df["year"] >= 2000].copy()
X_test = filtered_df[feature_cols]
y_test = filtered_df["co2"]
years = filtered_df["year"]

# === Generate charts and JSON per model ===
for fname in os.listdir(model_dir):
    if fname.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, fname)
        model_name = fname.replace("_best_pipeline.joblib", "")
        pipeline = joblib.load(model_path)

        # === Predictions ===
        y_pred = pipeline.predict(X_test)

        # === Line Chart ===
        plt.figure(figsize=(8, 5))
        plt.plot(years, y_test, label="Actual", marker='o', color="#3b82f6")
        plt.plot(years, y_pred, label="Predicted", marker='o', linestyle="--", color="#10b981")
        plt.title(f"{model_name.capitalize()} Prediction Accuracy (Years ≥ 2000)")
        plt.xlabel("Year")
        plt.ylabel("CO₂ Emissions (MT)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_accuracy.png")
        plt.close()

        # === Prediction JSON ===
        prediction_records = [
            {"year": int(y), "actual": round(float(a), 3), "predicted": round(float(p), 3)}
            for y, a, p in zip(years, y_test, y_pred)
        ]
        with open(f"{output_dir}/{model_name}_accuracy.json", "w") as f:
            json.dump(prediction_records, f, indent=2)

        # === Feature Importances ===
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            importances = pipeline.named_steps["model"].feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values(by="importance", ascending=False)

            # Save importance chart
            plt.figure(figsize=(9, 5))
            plt.barh(importance_df["feature"], importance_df["importance"], color="#6366f1")
            plt.gca().invert_yaxis()
            plt.title(f"{model_name.capitalize()} Feature Importances")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model_name}_importances.png")
            plt.close()

            # Save importance JSON
            importance_json = {
                row["feature"]: round(row["importance"], 4)
                for _, row in importance_df.iterrows()
            }
            with open(f"{output_dir}/{model_name}_importances.json", "w") as f:
                json.dump(importance_json, f, indent=2)

print("Accuracy + Feature importance PNG + JSON saved to:", output_dir)
