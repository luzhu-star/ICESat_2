import pandas as pd
import subprocess
from io import StringIO
import os
import geopandas as gpd
import numpy as np
import json

# Get the directory path where the current script is located
current_directory = os.path.dirname(os.path.abspath(__file__))
output_directory = os.path.join(current_directory, 'Data')

# Orbit data
orbit_data = [
   {"target_date": "2024-11-30", "target_rgt": "598"},# Replace with actual date and RGT

]

# Loop through each orbit
# Bounding box coordinates for your research area
bbox_coords = [
    [100.0, 15.0],  # Bottom-left corner (longitude, latitude)
    [100.0, 20.0],  # Top-left corner (longitude, latitude)
    [105.0, 20.0],  # Top-right corner (longitude, latitude)
    [105.0, 15.0],  # Bottom-right corner (longitude, latitude)
    [100.0, 15.0]   # Closing the loop back to the bottom-left corner
]
bbox_coords_str = json.dumps(bbox_coords)
df_listall = []
for orbit in orbit_data:
    target_date = orbit["target_date"]
    target_rgt = orbit["target_rgt"] 
    command = ["python", "icesat2.py", "--target_date=" + target_date, "--target_rgt=" + str(target_rgt), "--bbox_coords=" +  bbox_coords_str]
    result = subprocess.run(command, capture_output=True, text=True)

    # Handle errors if any
    if result.stderr:
        print(f"Error processing {target_date}, {target_rgt}: {result.stderr}")
        continue
    
    # If data is returned, read it into a DataFrame
    if result.stdout:
        df = pd.read_csv(StringIO(result.stdout))
        df_listall.append(df)

# Combine all DataFrames if available
if df_listall:  
    df_combined = pd.concat(df_listall, ignore_index=True)
else:
    print("No data frames were added for concatenation.")

# Save the combined DataFrame to an Excel file
output_file = os.path.join(output_directory, 'ICESAT_all.xlsx')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
df_combined.to_excel(output_file, index=False)

# Assuming you have the correct CRS definition
df_combined['longitude'] = df_combined.geometry.x
df_combined['latitude'] = df_combined.geometry.y
df_geo = df_combined[['longitude', 'latitude', 'depth']]

# Create a GeoDataFrame and set the CRS to EPSG:4326
gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.longitude, df_geo.latitude))
gdf.crs = "<EPSG:4326>"  

# Save the GeoDataFrame as a shapefile
output_shp_file = os.path.join(output_directory, 'ICESAT_all.shp')
gdf.to_file(output_shp_file)
