# Visualize model performance on test data with ROC curve and feature importance

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
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
y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1

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

# Step 8: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Step 9: Feature Importance (Logistic Regression Coefficients)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



























































# # Visualize model performance on test data with ROC curve

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
# from sklearn.model_selection import train_test_split

# # Step 1: Load the dataset
# df = pd.read_csv("final_irrigation_data.csv")
# bool_cols = df.select_dtypes(include='bool').columns
# df[bool_cols] = df[bool_cols].astype(int)

# # Step 2: Prepare features and labels
# X = df.drop(['water_given_ml', 'water_needed'], axis=1)
# y = df['water_needed']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 3: Load trained model
# model = joblib.load("irrigation_logistic_model.pkl")

# # Step 4: Predict
# y_pred = model.predict(X_test)
# y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1

# # Step 5: Accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f"✅ Model Accuracy: {accuracy:.2f}")

# # Step 6: Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # Step 7: Classification Report
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))

# # Step 8: ROC Curve
# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(fpr, tpr)

# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
