import pandas as pd

# Load the CSV file
df = pd.read_csv("final_irrigation_data.csv")  # replace with your actual file name

# Show first 10 rows (with all columns)
print(df.head(10))
