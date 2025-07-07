import firebase_admin
from firebase_admin import credentials, db
import joblib
import time

# 1. Firebase Admin Setup
cred = credentials.Certificate("firebase-adminsdk.json")  # Replace with your downloaded file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://aquagreen-819f3-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# 2. Load ML Model
model = joblib.load('irrigation_logistic_model_1.pkl')  # Make sure this file is in the same folder

# 3. Define Firebase Paths
sensor_ref = db.reference('/sensor_data')
prediction_ref = db.reference('/prediction')

# 4. Define Logic
def predict_and_update():
    data = sensor_ref.get()

    if not data:
        print("No sensor data found.")
        return

    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])

        prediction = model.predict([[temp, humidity]])[0]

        # Optional: convert prediction to label
        if prediction == 0:
            status = "Dry 🌵 - Water Needed"
        elif prediction == 1:
            status = "Ideal 🌿 - No Action"
        else:
            status = "Wet 💧 - Avoid Watering"

        # Send to Firebase
        prediction_ref.set({
            'moisture_prediction': int(prediction),
            'status': status
        })

        print(f"[PREDICTED] Temp: {temp}°C, Humidity: {humidity}%, Prediction: {status}")

    except Exception as e:
        print("Error during prediction:", e)

# 5. Loop Every 10 Seconds
while True:
    predict_and_update()
    time.sleep(10)