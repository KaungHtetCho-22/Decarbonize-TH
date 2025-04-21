import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import optuna
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
from joblib import dump
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# === Load datasets ===
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

# === Metadata and target ===
metadata_cols = ['country', 'year']
target_col = 'co2'

feature_cols = [
    'population', 'gdp', 'primary_energy_consumption',
    'oil_co2', 'coal_co2', 'cement_co2',
    'total_ghg', 'co2_including_luc',
    'temperature_change_from_ghg'
]

# === Log transform columns ===
log_transform_cols = [
    'population', 'gdp', 'primary_energy_consumption',
    'oil_co2', 'coal_co2', 'cement_co2',
    'total_ghg', 'co2_including_luc'
]

for df in [train_df, val_df, test_df]:
    for col in log_transform_cols:
        df[col] = np.log1p(df[col])

X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# === Feature scaling ===
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, feature_cols)
], remainder='drop')

# === Target scaler ===
os.makedirs("artifacts", exist_ok=True)
target_scaler = StandardScaler()
target_scaler.fit(y_train.values.reshape(-1, 1))
dump(target_scaler, "artifacts/target_scaler.pkl")

# Save log-transformed columns (optional for reproducibility)
with open("artifacts/log_transform_cols.txt", "w") as f:
    f.write("\n".join(log_transform_cols))

# === Model configs ===
model_configs = {
    "ridge": {"model": Ridge(), "param_ranges": {"alpha": (0.001, 100, 'log')}},
    "lasso": {"model": Lasso(), "param_ranges": {"alpha": (0.001, 10, 'log')}},
    "elastic_net": {"model": ElasticNet(), "param_ranges": {"alpha": (0.001, 10, 'log'), "l1_ratio": (0, 1)}},
    "random_forest": {"model": RandomForestRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "max_depth": (3, 20)}},
    "gradient_boosting": {"model": GradientBoostingRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "xgboost": {"model": xgb.XGBRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "lightgbm": {"model": lgbm.LGBMRegressor(random_state=42), "param_ranges": {"n_estimators": (50, 500), "learning_rate": (0.01, 0.3)}},
    "catboost": {"model": cb.CatBoostRegressor(random_state=42, verbose=0), "param_ranges": {"iterations": (50, 500), "learning_rate": (0.01, 0.3)}}
}

# === Evaluation function ===
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
mlflow.set_experiment("co2-forecasting-pipeline-log-scaled")
os.makedirs("models", exist_ok=True)

best_rmse = float('inf')
best_model_name = None
best_pipeline = None

for name, config in model_configs.items():
    print(f"\nTraining {name}...")
    with mlflow.start_run(run_name=name):
        y_train_scaled = target_scaler.transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

        pipeline, best_params, val_rmse = optimize_hyperparameters(
            name, config["model"], config["param_ranges"],
            X_train, y_train_scaled, X_val, y_val_scaled
        )

        pipeline.fit(X_train, y_train_scaled)
        y_train_pred = target_scaler.inverse_transform(pipeline.predict(X_train).reshape(-1, 1)).ravel()
        y_val_pred = target_scaler.inverse_transform(pipeline.predict(X_val).reshape(-1, 1)).ravel()
        y_test_pred = target_scaler.inverse_transform(pipeline.predict(X_test).reshape(-1, 1)).ravel()

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

        if test_metrics["test_rmse"] < best_rmse:
            best_rmse = test_metrics["test_rmse"]
            best_model_name = name
            best_pipeline = pipeline
            dump(pipeline, "models/best_model_pipeline.joblib")

        print(f"{name} done — Test RMSE: {test_metrics['test_rmse']:.4f}")

print(f"\nBest model: {best_model_name} with Test RMSE: {best_rmse:.4f}")