import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# === Load test data ===
test_df = pd.read_csv("data/processed/test_clean.csv")

# === Define columns ===
feature_cols = [
    'year', 'population', 'gdp',
    'cement_co2_per_capita', 'co2_growth_abs', 'co2_including_luc_growth_abs',
    'co2_including_luc_per_gdp', 'co2_including_luc_per_unit_energy',
    'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2_per_capita',
    'energy_per_capita', 'flaring_co2_per_capita',
    'nitrous_oxide_per_capita', 'temperature_change_from_n2o'
]

log_transform_cols = ['population', 'gdp']

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
os.makedirs("feature_importances", exist_ok=True)

for fname in os.listdir(model_dir):
    if fname.endswith("_best_pipeline.joblib"):
        model_path = os.path.join(model_dir, fname)
        model = joblib.load(model_path)
        model_name = fname.replace("_best_pipeline.joblib", "")

        y_pred = model.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        results.append({
            "model": model_name,
            "rmse": round(metrics["rmse"], 4),
            "mae": round(metrics["mae"], 4),
            "r2": round(metrics["r2"], 4),
            "mape": round(metrics["mape"], 4)
        })

        # === Save prediction plot ===
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.xlabel("True CO₂")
        plt.ylabel("Predicted CO₂")
        plt.title(f"{model_name} Predictions vs True Values")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.tight_layout()
        plt.savefig(f"feature_importances/{model_name}_test_predictions.png")
        plt.close()

# === Display results ===
results_df = pd.DataFrame(results).sort_values("rmse")
print("\n=== Inference Evaluation on Test Set: All Models ===")
print(results_df.to_string(index=False))
