import pandas as pd

# Load the original dataset
input_file = r"D:\gda_proj\data\processed_data_with_coords_cleaned.csv"
df = pd.read_csv(input_file)

# Function to remove 'e' notation by converting to standard float format
def convert_to_float(value):
    try:
        # Convert to float
        return float(value)
    except ValueError:
        # If conversion fails, return NaN
        return float('nan')

# Apply the function to the Latitude and Longitude columns to ensure proper float format
df['Latitude'] = df['Latitude'].apply(convert_to_float)
df['Longitude'] = df['Longitude'].apply(convert_to_float)

# Remove 'X' and 'Y' columns
df = df.drop(columns=['X', 'Y'])

# Reorder columns to move Latitude and Longitude to the front
df = df[['Latitude', 'Longitude'] + [col for col in df.columns if col not in ['Latitude', 'Longitude']]]

# Save the final cleaned dataset with a specific float format to avoid scientific notation
output_file = r"D:\gda_proj\data\processed_data_with_coords_final.csv"
df.to_csv(output_file, index=False, float_format='%.8f')

print(f"File saved as {output_file}")
