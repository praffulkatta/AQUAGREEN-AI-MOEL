# Realtime prediction using the trained irrigation model

import joblib

# Step 1: Load trained model
model = joblib.load("irrigation_logistic_model_1.pkl")

# Step 2: Get real-time input (simulate sensor)
soil_moisture = float(input("Enter current soil moisture (%): "))
temperature = float(input("Enter current temperature (°C): "))

# Step 3: Make prediction
input_data = [[soil_moisture, temperature]]
prediction = model.predict(input_data)[0]

# Step 4: Show result
if prediction == 1:
    print("🟢 Prediction: POUR water 💧")
else:
    print("🔴 Prediction: DO NOT pour water ❌")
