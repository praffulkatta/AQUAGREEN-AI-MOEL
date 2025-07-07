

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load your CSV file
df = pd.read_csv("soil_water_prediction_data.csv")  # ✅ Use your actual CSV file name

# Step 2: Define features and target
X = df[["soil_moisture", "temperature"]]  # ✅ Features
y = df["water_pour"]                      # ✅ Target label

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(" Logistic Regression Model Evaluation")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Step 6: Save the trained model
joblib.dump(model, "irrigation_logistic_model_1.pkl")
print(" Model saved as 'irrigation_logistic_model.pkl'")
