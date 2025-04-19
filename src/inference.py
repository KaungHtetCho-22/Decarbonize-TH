# import pandas as pd
# import numpy as np
# from joblib import load
# import os

# # Load target scaler and model
# target_scaler = load("artifacts/target_scaler.pkl")
# feature_scaler = load("artifacts/feature_scaler.pkl")
# model = load("models/lightgbm_best_model.joblib")

# # Define input features
# feature_cols = [
#     'population',
#     'gdp',
#     'primary_energy_consumption',
#     'oil_co2',
#     'coal_co2',
#     'total_ghg',
#     'co2_including_luc',
#     'temperature_change_from_ghg'
# ]

# # Create new input data as a DataFrame
# new_data = pd.DataFrame([{
#     'population': 71702438,
#     'gdp': 1124143726592,
#     'primary_energy_consumption': 1390.812,
#     'oil_co2': 104.343,
#     'coal_co2': 59.327,
#     'total_ghg': 416.852,
#     'co2_including_luc': 297.369,
#     'temperature_change_from_ghg': 0.015
# }])

# new_data_scaled = feature_scaler.transform(new_data[feature_cols])
# print(type(new_data))

# pred_scaled = model.predict(new_data_scaled)

# pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))

# print(f"Predicted CO₂ for Thailand in 2027: {pred_real[0][0]:,.2f}")

# import pandas as pd
# from joblib import load

# # Load the full pipeline (preprocessor + model)
# model = load("models/xgboost_best_model.joblib")

# # Load target scaler for inverse transform
# target_scaler = load("artifacts/target_scaler.pkl")

# # Raw input data
# new_data = pd.DataFrame([{
#     'population': 71702438,
#     'gdp': 1124143726592,
#     'primary_energy_consumption': 1390.812,
#     'oil_co2': 104.343,
#     'coal_co2': 59.327,
#     'total_ghg': 416.852,
#     'co2_including_luc': 297.369,
#     'temperature_change_from_ghg': 0.015
# }])

# # Run inference directly (preprocessing handled internally)
# pred_scaled = model.predict(new_data)
# pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))

# print(f"Predicted CO₂ for Thailand in 2027: {pred_real[0][0]:,.2f}")


import pandas as pd
from joblib import load

model = load("models/xgboost_best_pipeline.joblib")
target_scaler = load("artifacts/target_scaler.pkl")

for pop1 in [717, 717000, 71700000]:
    new_data = pd.DataFrame([{
        'population': pop1,
        'gdp': 1124143726592,
        'primary_energy_consumption': 1390.812,
        'oil_co2': 104.343,
        'coal_co2': 59.327,
        'total_ghg': 416.852,
        'co2_including_luc': 297.369,
        'temperature_change_from_ghg': 0.015
    }])
    pred_scaled = model.predict(new_data)
    pred_real = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    print(f"Population {pop1:,}: Predicted CO₂ = {pred_real[0][0]:,.2f}")
