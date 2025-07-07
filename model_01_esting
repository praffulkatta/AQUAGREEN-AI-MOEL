# Visualize model performance on test data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
df = pd.read_csv("final_irrigation_data.csv")
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Step 2: Prepare features and labels
X = df.drop(['water_given_ml', 'water_needed'], axis=1)
y = df['water_needed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Load trained model
model = joblib.load("irrigation_logistic_model.pkl")

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Step 6: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
