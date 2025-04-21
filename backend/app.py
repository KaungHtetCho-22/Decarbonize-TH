from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
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

model = joblib.load("best_model_pipeline.joblib")

@app.post("/predict")
def predict(features: Features):
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
    prediction = model.predict(X)[0]
    return {"prediction": round(prediction, 2)}
