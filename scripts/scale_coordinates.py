import pandas as pd

# Load the dataset
input_file = r"D:\gda_proj\data\processed_data_with_coords_final.csv"
df = pd.read_csv(input_file)

# Define a scaling factor
scaling_factor = 100  # Try 100, and adjust higher if needed

# Apply scaling to Latitude and Longitude columns
df['Latitude'] = df['Latitude'] * scaling_factor
df['Longitude'] = df['Longitude'] * scaling_factor

# Save the modified dataset with scaled coordinates
output_file = r"D:\gda_proj\data\processed_data_with_coords_scaled.csv"
df.to_csv(output_file, index=False, float_format='%.8f')

print(f"File saved with scaled coordinates as {output_file}")
