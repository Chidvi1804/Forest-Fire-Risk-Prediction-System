import pandas as pd

# Load the processed dataset
data = pd.read_csv("../data/processed_data.csv")  # Adjust the path to point to your CSV

# Display the first few rows and the columns of the dataset
print("First few rows of the dataset:")
print(data.head())
print("\nColumns in the dataset:")
print(data.columns)
