from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend connection (React on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = load("models/catboost_best_pipeline.joblib")
target_scaler = load("artifacts/target_scaler.pkl")

# Define input schema
class PredictRequest(BaseModel):
    population: float
    gdp: float
    primary_energy_consumption: float
    oil_co2: float
    coal_co2: float
    total_ghg: float
    co2_including_luc: float
    temperature_change_from_ghg: float

@app.post("/predict")
def predict(data: PredictRequest):
    input_df = pd.DataFrame([data.dict()])
    pred_scaled = model.predict(input_df)
    pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    return {"prediction": float(pred_real[0][0])}
