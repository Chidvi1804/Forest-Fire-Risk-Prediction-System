import pandas as pd

# Load the dataset using the absolute path
data = pd.read_csv(r"C:\Users\csrid\OneDrive\Desktop\gda_proj\data\fire_data_original.csv.csv")  # Use r to handle backslashes

# Select relevant features
features = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
data = data[features + ['y']]  # Add 'y' for target variable

# Save the processed dataset
data.to_csv(r"C:\Users\csrid\OneDrive\Desktop\gda_proj\data\processed_data.csv", index=False)

print("Processed dataset saved to 'data/processed_data.csv'")
