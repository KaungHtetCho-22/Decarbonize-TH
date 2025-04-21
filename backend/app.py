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
    year: int
    population: float
    gdp: float
    cement_co2_per_capita: float
    co2_growth_abs: float
    co2_including_luc_growth_abs: float
    co2_including_luc_per_gdp: float
    co2_including_luc_per_unit_energy: float
    co2_per_gdp: float
    co2_per_unit_energy: float
    coal_co2_per_capita: float
    energy_per_capita: float
    flaring_co2_per_capita: float
    nitrous_oxide_per_capita: float
    temperature_change_from_n2o: float

log_transform_cols = ["population", "gdp"]

feature_cols = [
    'year', 'population', 'gdp',
    'cement_co2_per_capita', 'co2_growth_abs', 'co2_including_luc_growth_abs',
    'co2_including_luc_per_gdp', 'co2_including_luc_per_unit_energy',
    'co2_per_gdp', 'co2_per_unit_energy', 'coal_co2_per_capita',
    'energy_per_capita', 'flaring_co2_per_capita',
    'nitrous_oxide_per_capita', 'temperature_change_from_n2o'
]

# Load pipeline (includes scaler + model)
model = joblib.load("xgboost_best_pipeline.joblib")

@app.post("/predict")
def predict(features: Features):
    # Convert input to DataFrame
    X = pd.DataFrame([features.dict()])

    # Apply log1p to selected columns
    for col in log_transform_cols:
        X[col] = np.log1p(X[col])

    # Predict
    y_pred = model.predict(X)

    return {"prediction": round(float(y_pred[0]), 2)}
