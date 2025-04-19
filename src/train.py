import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import mlflow
import mlflow.sklearn
from joblib import dump
import os
import warnings
warnings.filterwarnings('ignore')

# === Set MLflow experiment ===
mlflow.set_experiment("co2-forecasting-production")

# === Load data from separate files ===
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Separate metadata (for logging purposes) from features
metadata_cols = []
for col in ['country', 'year']:
    if col in train_df.columns:
        metadata_cols.append(col)

target_col = 'co2'

# Prepare datasets
X_train = train_df.drop(columns=[target_col] + metadata_cols)
y_train = train_df[target_col]

X_val = val_df.drop(columns=[target_col] + metadata_cols)
y_val = val_df[target_col]

X_test = test_df.drop(columns=[target_col] + metadata_cols)
y_test = test_df[target_col]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# === Feature columns by type (for preprocessing) ===
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# === Save scalers used in the pipeline ===
# This will be the same scaler used in the preprocessing pipeline
scaler = RobustScaler()
scaler.fit(X_train[num_cols])
os.makedirs("artifacts", exist_ok=True)
dump(scaler, "artifacts/feature_scaler.pkl")

target_scaler = StandardScaler()
target_scaler.fit(y_train.values.reshape(-1, 1))
dump(target_scaler, "artifacts/target_scaler.pkl")

# === Helper function for evaluation metrics ===
def evaluate_model(model, X, y_true, prefix=""):
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = np.nan
    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape
    }
    return metrics, y_pred

# === Define preprocessing pipelines ===
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', scaler)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
    ],
    remainder='drop'
)

# === Define model architectures and ranges for hyperparameter tuning ===
model_configs = {
    "ridge": {
        "model": Ridge(),
        "param_ranges": {
            "alpha": (0.001, 100, 'log')
        }
    },
    "lasso": {
        "model": Lasso(),
        "param_ranges": {
            "alpha": (0.001, 10, 'log')
        }
    },
    "elastic_net": {
        "model": ElasticNet(),
        "param_ranges": {
            "alpha": (0.001, 10, 'log'),
            "l1_ratio": (0, 1)
        }
    },
    "random_forest": {
        "model": RandomForestRegressor(random_state=42),
        "param_ranges": {
            "n_estimators": (50, 500),
            "max_depth": (3, 20),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10)
        }
    },
    "gradient_boosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "param_ranges": {
            "n_estimators": (50, 500),
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "subsample": (0.5, 1.0)
        }
    },
    "xgboost": {
        "model": xgb.XGBRegressor(random_state=42),
        "param_ranges": {
            "n_estimators": (50, 500),
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0, 5),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5)
        }
    },
    "lightgbm": {
        "model": lgbm.LGBMRegressor(random_state=42),
        "param_ranges": {
            "n_estimators": (50, 500),
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "num_leaves": (10, 100),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "reg_alpha": (0, 5),
            "reg_lambda": (0, 5)
        }
    },
    "catboost": {
        "model": cb.CatBoostRegressor(random_state=42, verbose=0),
        "param_ranges": {
            "iterations": (50, 500),
            "learning_rate": (0.01, 0.3),
            "depth": (3, 10),
            "l2_leaf_reg": (1, 10)
        }
    }
}

# === Function to optimize hyperparameters using Optuna ===
def optimize_hyperparameters(model_name, model, param_ranges, X_train, y_train, X_val, y_val, n_trials=30):
    """Optimize hyperparameters using Optuna with validation data."""
    
    def objective(trial):
        params = {}
        
        # Create hyperparameter suggestions based on ranges
        for param, value_range in param_ranges.items():
            if len(value_range) == 3 and value_range[2] == 'log':
                params[param] = trial.suggest_float(param, value_range[0], value_range[1], log=True)
            elif isinstance(value_range, list):
                params[param] = trial.suggest_categorical(param, value_range)
            elif isinstance(value_range[0], int):
                params[param] = trial.suggest_int(param, value_range[0], value_range[1])
            else:
                params[param] = trial.suggest_float(param, value_range[0], value_range[1])
                
        # Set parameters
        for param, value in params.items():
            setattr(model, param, value)
            
        # Full pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        return rmse
    
    # Create and run optimization study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    # Apply best parameters to model
    for param, value in study.best_params.items():
        setattr(model, param, value)
    
    # Create and return pipeline with tuned model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline, study.best_params, study.best_value

# === Train, evaluate and log each model ===
best_models = {}
best_rmse = float('inf')
best_model_name = None

for model_name, config in model_configs.items():
    print(f"\n🔧 Training: {model_name}")
    
    with mlflow.start_run(run_name=model_name):
        # Optimize hyperparameters
        model, best_params, val_rmse = optimize_hyperparameters(
            model_name,
            config["model"],
            config["param_ranges"],
            X_train, y_train, X_val, y_val,
            n_trials=25  # Adjust based on computational resources
        )
        
        # Train with optimized hyperparameters
        model.fit(X_train, y_train)
        
        # Evaluate on train, validation, and test sets
        train_metrics, y_train_pred = evaluate_model(model, X_train, y_train, "train_")
        val_metrics, y_val_pred = evaluate_model(model, X_val, y_val, "val_")
        test_metrics, y_test_pred = evaluate_model(model, X_test, y_test, "test_")
        
        # Combine all metrics
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        
        # Log parameters and metrics to MLflow
        mlflow.log_params(best_params)
        for name, value in all_metrics.items():
            mlflow.log_metric(name, value)
        
        # Log feature importance if available
        try:
            if hasattr(model.named_steps['model'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.named_steps['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save feature importance plot
                plt.figure(figsize=(10, 6))
                plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
                plt.xlabel('Importance')
                plt.title(f'Top 15 Feature Importances - {model_name}')
                importance_path = f"feature_importance_{model_name}.png"
                plt.tight_layout()
                plt.savefig(importance_path)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)  # Clean up
                
                # Log as CSV
                feature_importance.to_csv(f"feature_importance_{model_name}.csv", index=False)
                mlflow.log_artifact(f"feature_importance_{model_name}.csv")
                os.remove(f"feature_importance_{model_name}.csv")  # Clean up
        except:
            print(f"Could not log feature importance for {model_name}")
        
        # Save prediction vs. actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs. Predicted - {model_name}')
        residual_path = f"residual_plot_{model_name}.png"
        plt.savefig(residual_path)
        mlflow.log_artifact(residual_path)
        os.remove(residual_path)  # Clean up
        
        # Input example for schema tracking
        input_example = X_train.iloc[:1]
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )
        
        # Register model in the model registry if it's the best so far
        test_rmse = all_metrics["test_rmse"]
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model_name = model_name
            
            # Register model in MLflow Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            try:
                # Try to register the model (requires MLflow tracking server)
                model_details = mlflow.register_model(model_uri, f"CO2Forecasting-{model_name}")
                mlflow.log_param("registered_model_version", model_details.version)
            except:
                print("Model registry not available - skipping registration")
            
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}_best_model.joblib"
        dump(model, model_path)

        # Log scalers to MLflow
        mlflow.log_artifact("artifacts/feature_scaler.pkl", artifact_path="preprocessing")
        mlflow.log_artifact("artifacts/target_scaler.pkl", artifact_path="preprocessing")

        
        print(f"Model: {model_name}")
        print(f"Train RMSE: {all_metrics['train_rmse']:.4f}, R²: {all_metrics['train_r2']:.4f}")
        print(f"Val RMSE: {all_metrics['val_rmse']:.4f}, R²: {all_metrics['val_r2']:.4f}")
        print(f"Test RMSE: {all_metrics['test_rmse']:.4f}, R²: {all_metrics['test_r2']:.4f}")
        
        # Store best model for ensemble creation
        best_models[model_name] = model

# === Create a stacking ensemble from top models ===
print("\n🔧 Training: Stacking Ensemble")
with mlflow.start_run(run_name="stacking_ensemble"):
    # Create stacking ensemble
    stacking_model = StackingRegressor(
        estimators=[(name, model) for name, model in best_models.items()],
        final_estimator=Ridge(),
        cv=5
    )
    
    # Fit stacking model
    stacking_model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics, _ = evaluate_model(stacking_model, X_train, y_train, "train_")
    val_metrics, _ = evaluate_model(stacking_model, X_val, y_val, "val_")
    test_metrics, _ = evaluate_model(stacking_model, X_test, y_test, "test_")
    
    # Combine metrics
    all_metrics = {**train_metrics, **val_metrics, **test_metrics}
    
    # Log metrics
    for name, value in all_metrics.items():
        mlflow.log_metric(name, value)
    
    # Log and register model
    input_example = X_train.iloc[:1]
    mlflow.sklearn.log_model(stacking_model, 
                             artifact_path="model",
                             input_example=input_example)
    
    # Register model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
    try:
        # Try to register the model (requires MLflow tracking server)
        model_details = mlflow.register_model(model_uri, "CO2Forecasting-StackingEnsemble")
    except:
        print("Model registry not available - skipping registration")
    
    # Save model locally
    model_path = f"models/stacking_ensemble_best_model.joblib"
    dump(stacking_model, model_path)
    
    print(f"Model: Stacking Ensemble")
    print(f"Train RMSE: {all_metrics['train_rmse']:.4f}, R²: {all_metrics['train_r2']:.4f}")
    print(f"Val RMSE: {all_metrics['val_rmse']:.4f}, R²: {all_metrics['val_r2']:.4f}")
    print(f"Test RMSE: {all_metrics['test_rmse']:.4f}, R²: {all_metrics['test_r2']:.4f}")

print(f"\n Best performing model: {best_model_name} with Test RMSE: {best_rmse:.4f}")