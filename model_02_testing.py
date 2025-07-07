# Testing and Visualization for Irrigation Prediction Model

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load your CSV and model
df = pd.read_csv("soil_water_prediction_data.csv")
model = joblib.load("irrigation_logistic_model_1.pkl")

# Step 2: Prepare features and target
X = df[["soil_moisture", "temperature"]]
y = df["water_pour"]

# Step 3: Predict with the trained model
y_pred = model.predict(X)

# Step 4: Show accuracy
accuracy = accuracy_score(y, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("\n🧾 Confusion Matrix:\n", conf_matrix)
print("\n📊 Classification Report:\n", classification_report(y, y_pred))

# Step 6: Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Water", "Water"], yticklabels=["No Water", "Water"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Step 7: Optional – Scatter plot of prediction
plt.figure(figsize=(7, 5))
plt.scatter(df["soil_moisture"], df["temperature"], c=y_pred, cmap='coolwarm', edgecolors='k')
plt.xlabel("Soil Moisture (%)")
plt.ylabel("Temperature (°C)")
plt.title("Prediction Scatter: Red=No Water, Blue=Water")
plt.colorbar(label="Prediction (0=No, 1=Yes)")
plt.grid(True)
plt.tight_layout()
plt.show()
