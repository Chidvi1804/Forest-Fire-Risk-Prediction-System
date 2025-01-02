import pandas as pd
import numpy as np

# Load the dataset
input_file = r"D:\gda_proj\data\processed_data_with_coords_final.csv"
df = pd.read_csv(input_file)

# Jitter function to add a small random adjustment
def add_jitter(value, scale=1e-4):
    return value + np.random.uniform(-scale, scale)

# Apply jitter to Latitude and Longitude columns
df['Latitude'] = df['Latitude'].apply(lambda x: add_jitter(x, scale=1e-5))  # Adjust scale as needed
df['Longitude'] = df['Longitude'].apply(lambda x: add_jitter(x, scale=1e-5))

# Save the new dataset with jittered coordinates
output_file = r"D:\gda_proj\data\processed_data_with_coords_jittered.csv"
df.to_csv(output_file, index=False, float_format='%.8f')

print(f"File saved with jittered coordinates as {output_file}")
