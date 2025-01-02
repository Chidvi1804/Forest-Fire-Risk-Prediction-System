import pandas as pd

# Load the jittered data
file_path = r"D:/gda_proj/data/processed_data_with_coords_jittered.csv"
df = pd.read_csv(file_path)

# Function to remove 'e' notation by converting to standard float format
def convert_to_float(value):
    try:
        # Convert to float
        return float(value)
    except ValueError:
        # If conversion fails, return NaN
        return float('nan')

# Apply the function to Latitude and Longitude columns to handle 'e' notation
df['Latitude'] = df['Latitude'].apply(convert_to_float)
df['Longitude'] = df['Longitude'].apply(convert_to_float)

# Apply scaling for latitude and longitude
scale_factor_lat = 10  # Adjust this value as needed
scale_factor_lon = 10  # Adjust this value as needed
df['scaled_lat'] = df['Latitude'] * scale_factor_lat
df['scaled_lon'] = df['Longitude'] * scale_factor_lon

# Add a unique Point_ID column
df['Point_ID'] = df.index  # This will give each row a unique ID based on its index

# Drop the original Latitude and Longitude columns
df = df.drop(columns=['Latitude', 'Longitude'])

# Reorder columns to place scaled_lat and scaled_lon first, followed by other columns and Point_ID last
columns_order = ['scaled_lat', 'scaled_lon', 'Point_ID'] + [col for col in df.columns if col not in ['scaled_lat', 'scaled_lon', 'Point_ID']]
df = df[columns_order]

# Save the modified dataset with scaled coordinates and without scientific notation
df.to_csv(r"D:/gda_proj/data/processed_data_with_coords_jittered_scaled_with_id.csv", index=False, float_format='%.8f')

# Check the first few rows to verify the final structure
print(df.head())
