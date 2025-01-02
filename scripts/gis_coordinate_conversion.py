import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj, transform

# Define projection system for input data (UTM Zone 29N, adjust if needed)
input_proj = Proj(init='epsg:32629')  # Assuming UTM Zone 29N
# Define WGS84 projection (lat/lon)
output_proj = Proj(init='epsg:4326')  # WGS84 for latitude/longitude

# Load your dataset from the specified path
data = pd.read_csv(r"D:\gda_proj\data\processed_data.csv")

# Function to convert 'X' and 'Y' coordinates to latitude and longitude
def convert_coordinates(row):
    # Convert (X, Y) to (lat, lon)
    lon, lat = transform(input_proj, output_proj, row['X'], row['Y'])
    return pd.Series([lat, lon], index=['Latitude', 'Longitude'])

# Apply the conversion function to each row and create new columns for latitude and longitude
data[['Latitude', 'Longitude']] = data.apply(convert_coordinates, axis=1)

# Save the updated data with Latitude and Longitude columns
data.to_csv(r"D:\gda_proj\data\processed_data_with_coords.csv", index=False)

# Confirm successful conversion
print("Coordinate conversion complete. Saved to processed_data_with_coords.csv")
