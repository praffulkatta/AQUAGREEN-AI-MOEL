import pandas as pd
import numpy as np

# Step 1: Load your CSV file
df = pd.read_csv("irrigation_data.csv")

# Step 2: Check basic info
print("Before cleaning:\n", df.info())
print("\nMissing values:\n", df.isnull().sum())

# Step 3: Drop rows with missing or bad values (if any)
df = df.dropna()

# Step 4: Remove outliers (optional, based on your data)
df = df[(df['soil_moisture'] >= 0) & (df['soil_moisture'] <= 100)]
df = df[(df['temperature'] >= -10) & (df['temperature'] <= 60)]
df = df[(df['air_humidity'] >= 0) & (df['air_humidity'] <= 100)]
df = df[(df['wind_speed'] >= 0)]

# Step 5: Normalize (scale) numeric features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cols_to_scale = ['soil_moisture', 'temperature', 'air_humidity', 'wind_speed']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Step 6: Feature Engineering - add high-level features
# Example: Heat Index (combines temp & humidity)
df['heat_index'] = 0.5 * (df['temperature'] + df['air_humidity'])

# Example: Drought flag
df['is_dry'] = (df['soil_moisture'] < 0.3).astype(int)  # scaled threshold

# Example: Time of Day Group
df['time_group'] = pd.cut(df['time_of_day'], bins=[-1,5,11,17,21,24], labels=["Night", "Morning", "Afternoon", "Evening", "Late Night"])

# Step 7: One-hot encode categorical features (like rain, time_group)
df = pd.get_dummies(df, columns=['rain_next_hour', 'time_group'], drop_first=True)

# Step 8: Save cleaned file
df.to_csv("cleaned_irrigation_data.csv", index=False)
print("✅ Cleaned & enriched dataset saved as 'cleaned_irrigation_data.csv'")
