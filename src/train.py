import pandas as pd
import numpy as np
import os
import optuna
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
from joblib import dump
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# === Load datasets ===
train_df = pd.read_csv("data/processed/train_clean.csv")
val_df = pd.read_csv("data/processed/val_clean.csv")
test_df = pd.read_csv("data/processed/test_clean.csv")

# === Feature and Target Columns ===
feature_cols = [
    'year', 'population', 'gdp',
    'cement_co2_per_capita', 'co2_growth_abs', 'co2_including_luc_growth_abs',
    'co2_including_luc_per_gdp', 'co2_including_luc_per_unit_energy',
    'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2_per_capita',
    'energy_per_capita', 'flaring_co2_per_capita',
    'nitrous_oxide_per_capita', 'temperature_change_from_n2o'
]
target_col = 'co2'

log_transform_cols = ['population', 'gdp']
for df in [train_df, val_df, test_df]:
    for col in log_transform_cols:
        df[col] = np.log1p(df[col])

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# === Feature Scaling Pipeline ===
num_pipeline = Pipeline([
    ("scaler", StandardScaler())
])


preprocessor = ColumnTransformer([
    ("num", num_pipeline, feature_cols)
], remainder='drop')

# === Model configs ===
model_configs = {
    "random_forest": {"model": RandomForestRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "max_depth": (3, 20)}},
    "gradient_boosting": {"model": GradientBoostingRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "xgboost": {"model": xgb.XGBRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "lightgbm": {"model": lgbm.LGBMRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "catboost": {"model": cb.CatBoostRegressor(random_state=42, verbose=0), "param_ranges": {"iterations": (50, 500), "learning_rate": (0.01, 0.3)}}
}

# === Evaluation ===
def evaluate_model(y_true, y_pred, prefix=""):
    return {
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
        f"{prefix}mape": mean_absolute_percentage_error(y_true, y_pred)
    }, y_pred

# === Optimization ===
def optimize_hyperparameters(name, model, param_ranges, X_train, y_train, X_val, y_val, n_trials=25):
    def objective(trial):
        params = {}
        for param, range_ in param_ranges.items():
            if len(range_) == 3 and range_[2] == 'log':
                params[param] = trial.suggest_float(param, range_[0], range_[1], log=True)
            elif isinstance(range_[0], int):
                params[param] = trial.suggest_int(param, range_[0], range_[1])
            else:
                params[param] = trial.suggest_float(param, range_[0], range_[1])
        for param, value in params.items():
            setattr(model, param, value)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    for param, value in study.best_params.items():
        setattr(model, param, value)
    return Pipeline([("preprocessor", preprocessor), ("model", model)]), study.best_params, study.best_value

# === Train & Save Models ===
mlflow.set_experiment("co2-forecasting-new-features")
os.makedirs("models", exist_ok=True)
os.makedirs("feature_importances", exist_ok=True)

best_rmse = float('inf')
best_model_name = None
best_pipeline = None

for name, config in model_configs.items():
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        pipeline, best_params, val_rmse = optimize_hyperparameters(
            name, config["model"], config["param_ranges"],
            X_train, y_train, X_val, y_val
        )

        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)

        train_metrics, _ = evaluate_model(y_train, y_train_pred, "train_")
        val_metrics, _ = evaluate_model(y_val, y_val_pred, "val_")
        test_metrics, _ = evaluate_model(y_test, y_test_pred, "test_")

        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_params(best_params)
        mlflow.log_param("log_transform", ",".join(log_transform_cols))
        for k, v in all_metrics.items():
            mlflow.log_metric(k, v)

        model_path = f"models/{name}_best_pipeline.joblib"
        dump(pipeline, model_path)

        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            importances = pipeline.named_steps["model"].feature_importances_
            importances_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values(by="importance", ascending=False)
            # importances_df.to_csv(f"feature_importances/{name}_importances.csv", index=False)

            plt.figure(figsize=(10, 5))
            plt.barh(importances_df.feature, importances_df.importance)
            plt.gca().invert_yaxis()
            plt.title(f"{name} Feature Importances")
            plt.tight_layout()
            plt.savefig(f"feature_importances/{name}_importances.png")
            plt.close()

        # Plot predictions vs actual
        plt.figure()
        plt.scatter(y_test, y_test_pred, alpha=0.7)
        plt.xlabel("True CO2")
        plt.ylabel("Predicted CO2")
        plt.title(f"{name} Predictions vs True Values")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        plt.tight_layout()
        plt.savefig(f"feature_importances/{name}_predictions.png")
        plt.close()

        if test_metrics["test_rmse"] < best_rmse:
            best_rmse = test_metrics["test_rmse"]
            best_model_name = name
            best_pipeline = pipeline
            dump(pipeline, "models/best_model_pipeline.joblib")

        print(f"{name} done — Test RMSE: {test_metrics['test_rmse']:.4f}")

print(f"\nBest model: {best_model_name} with Test RMSE: {best_rmse:.4f}")
