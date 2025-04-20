from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORS for local frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for security
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
    total_ghg: float
    co2_including_luc: float
    temperature_change_from_ghg: float

# Load your model
model = joblib.load("models/best_model_pipeline.joblib")  # ensure this path is correct inside Docker container

@app.post("/predict")
def predict(features: Features):
    X = [[
        features.population,
        features.gdp,
        features.primary_energy_consumption,
        features.oil_co2,
        features.coal_co2,
        features.total_ghg,
        features.co2_including_luc,
        features.temperature_change_from_ghg
    ]]
    prediction = model.predict(X)[0]
    return {"prediction": round(prediction, 2)}
