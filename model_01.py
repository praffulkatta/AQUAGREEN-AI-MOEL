# Logistic Regression Model for Irrigation Prediction
# Goal: Predict whether watering is needed (1) or not (0)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Step 1: Load the final dataset
df = pd.read_csv("final_irrigation_data.csv")  # Replace with your actual file name

# Step 2: Convert boolean columns to integers if needed
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Step 3: Define features and target for classification
X = df.drop(['water_given_ml', 'water_needed'], axis=1)  # Input features
y = df['water_needed']  # Target label (0 or 1)

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("✅ Logistic Regression Model Evaluation")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Step 7: Save the trained model
joblib.dump(model, "irrigation_logistic_model.pkl")
print("✅ Model saved as 'irrigation_logistic_model.pkl'")
