from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Features(BaseModel):
    population: float
    gdp: float
    primary_energy_consumption: float
    oil_co2: float
    coal_co2: float
    cement_co2: float
    total_ghg: float
    co2_including_luc: float
    temperature_change_from_ghg: float

log_transform_cols = [
    "population", "gdp", "primary_energy_consumption",
    "oil_co2", "coal_co2", "cement_co2",
    "total_ghg", "co2_including_luc"
]

feature_cols = log_transform_cols + ["temperature_change_from_ghg"]

model = joblib.load("xgboost_best_pipeline.joblib")  # Must match your frontend call

@app.post("/predict")
def predict(features: Features):
    # Create a DataFrame
    X = pd.DataFrame([{
        "population": features.population,
        "gdp": features.gdp,
        "primary_energy_consumption": features.primary_energy_consumption,
        "oil_co2": features.oil_co2,
        "coal_co2": features.coal_co2,
        "cement_co2": features.cement_co2,
        "total_ghg": features.total_ghg,
        "co2_including_luc": features.co2_including_luc,
        "temperature_change_from_ghg": features.temperature_change_from_ghg
    }])

    # Apply log1p to the necessary columns
    for col in log_transform_cols:
        X[col] = np.log1p(X[col])

    # Predict using the full pipeline (includes StandardScaler etc.)
    y_scaled = model.predict(X)
    target_scaler = joblib.load("target_scaler.pkl")  # make sure this exists in container
    y_pred = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

    return {"prediction": round(float(y_pred[0]), 2)}
