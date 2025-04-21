import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# === Load test data ===
test_df = pd.read_csv("data/processed/test.csv")

# === Load target scaler ===
target_scaler = joblib.load("artifacts/target_scaler.pkl")

# === Define columns ===
feature_cols = [
    'population', 'gdp', 'primary_energy_consumption',
    'oil_co2', 'coal_co2', 'cement_co2',
    'total_ghg', 'co2_including_luc',
    'temperature_change_from_ghg'
]

log_transform_cols = [
    'population', 'gdp', 'primary_energy_consumption',
    'oil_co2', 'coal_co2', 'cement_co2',
    'total_ghg', 'co2_including_luc'
]

# === Preprocess test data ===
for col in log_transform_cols:
    test_df[col] = np.log1p(test_df[col])
X_test = test_df[feature_cols]
y_test = test_df["co2"]

# === Evaluation function ===
def evaluate(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred)
    }

# === Evaluate each model ===
results = []
model_dir = "models"

for fname in os.listdir(model_dir):
    if fname.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, fname)
        model = joblib.load(model_path)
        model_name = fname.replace("_best_pipeline.joblib", "")

        y_pred_scaled = model.predict(X_test)
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        metrics = evaluate(y_test, y_pred)
        results.append({
            "model": model_name,
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"], 4),
            "r2": round(metrics["r2"], 4),
            "mape": round(metrics["mape"], 4)
        })

# === Display results ===
results_df = pd.DataFrame(results).sort_values("rmse")
print("\n=== Inference Evaluation on Test Set: All Models ===")
print(results_df.to_string(index=False))
