import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("cleaned_irrigation_data.csv")  # replace with actual filename

# Step 2: Convert boolean columns to integers (0/1)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Step 3: Confirm no missing values
assert df.isnull().sum().sum() == 0, "Data contains null values"

# Step 4: Split features and labels
# Option 1: For regression (predict how much water to give)
X = df.drop(['water_given_ml', 'water_needed'], axis=1)
y_regression = df['water_given_ml']

# Option 2: For classification (predict yes/no watering)
y_classification = df['water_needed']

# You can now use X + y_regression or X + y_classification

# Step 5: Save the cleaned version (optional)
df.to_csv("final_irrigation_data.csv", index=False)
print("✅ Final dataset saved. Ready for ML training.")
