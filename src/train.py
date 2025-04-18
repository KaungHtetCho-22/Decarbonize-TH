import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from joblib import dump
import os

# === Load data ===
df = pd.read_csv("data/processed/train.csv")

# Drop object columns (e.g., 'country', 'year' if present)
drop_cols = ['co2']
if 'country' in df.columns:
    drop_cols.append('country')
if 'year' in df.columns:
    drop_cols.append('year')

X = df.drop(columns=drop_cols)
y = df['co2']

# === Define models & hyperparameters ===
models = {
    "xgboost": (
        xgb.XGBRegressor(),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5]
        }
    ),
    "ridge": (
        Ridge(),
        {
            "model__alpha": [0.1, 1.0, 10]
        }
    ),
    "random_forest": (
        RandomForestRegressor(),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5]
        }
    )
}

# === Set MLflow experiment ===
mlflow.set_experiment("co2-forecasting")

# === Train and log each model ===
for model_name, (estimator, param_grid) in models.items():
    print(f"\n🔧 Training: {model_name}")
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline([
            ("model", estimator)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            error_score="raise"
        )
        grid.fit(X, y)

        best_model = grid.best_estimator_
        preds = best_model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)

        # === MLflow logging ===
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Input signature logging (for deployment)
        input_example = X.iloc[:1]
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            input_example=input_example
        )

        # === Save model locally ===
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}_best_model.joblib"
        dump(best_model, model_path)

        print(f"✅ Logged and saved: {model_name} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
