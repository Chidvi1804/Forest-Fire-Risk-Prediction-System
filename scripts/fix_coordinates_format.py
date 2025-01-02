import pandas as pd

# Load the dataset
file_path = r"D:\gda_proj\data\processed_data_with_coords.csv"
df = pd.read_csv(file_path)

# Function to remove 'e' notation by converting to standard float format
def convert_to_float(value):
    try:
        # Convert to float
        return float(value)
    except ValueError:
        # If conversion fails, return NaN
        return float('nan')

# Apply the function to the Latitude and Longitude columns
df['Latitude'] = df['Latitude'].apply(convert_to_float)
df['Longitude'] = df['Longitude'].apply(convert_to_float)

# Save the cleaned dataset with a specific float format to avoid scientific notation
df.to_csv(r"D:\gda_proj\data\processed_data_with_coords_cleaned.csv", index=False, float_format='%.8f')

# Check the first few rows to make sure the conversion was successful
print(df[['Latitude', 'Longitude']].head())
