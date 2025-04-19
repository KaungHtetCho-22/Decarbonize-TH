import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model_pipeline = joblib.load('models/lightgbm_best_model.joblib')

# Create a new unseen data sample - add the missing 'Unnamed: 0' column
new_data = pd.DataFrame([{
    'Unnamed: 0': 0,  # Add this column with a placeholder value
    'population': 71702438,
    'gdp': 1124143726592,
    'primary_energy_consumption': 1390.812,
    'oil_co2': 104.343,
    'coal_co2': 59.327,
    'total_ghg': 416.852,
    'co2_including_luc': 297.369,
    'temperature_change_from_ghg': 0.015
}])

# Make prediction
prediction = model_pipeline.predict(new_data)

# If the prediction is scaled, we need to inverse transform it
# Since we don't have the target_scaler saved, let's create one from original data if needed
try:
    # Check if prediction looks scaled (between 0 and 1)
    if 0 <= prediction[0] <= 1:
        print("Prediction appears to be scaled, attempting to inverse transform...")
        train_df = pd.read_csv('data/processed/train.csv')
        target_scaler = MinMaxScaler()
        target_scaler.fit(train_df[['co2']])
        prediction = target_scaler.inverse_transform(prediction.reshape(-1, 1))
        prediction = prediction[0][0]
except Exception as e:
    print(f"Note: {e}")

print(f"Predicted CO₂ using LightGBM model: {prediction:,.2f}")