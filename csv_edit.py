import pandas as pd
import re
import numpy as np
import math 

def convert_to_vector(longitude, latitude, azimuth):
    # Convert latitude and longitude to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    
    # Convert azimuth to radians
    az_rad = math.radians(azimuth)
    
    # Calculate x, y, z components of the vector
    x = math.cos(lat_rad) * math.cos(lon_rad) * math.cos(az_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad) * math.cos(az_rad)
    z = math.sin(lat_rad) * math.sin(az_rad)
    
    return x, y, z

def process_data_set(csv_path = "dataset.csv"):

    df = pd.read_csv(csv_path)

    # get cos(azimuth)
    df['cos_Main_Azimuth'] = np.cos(np.radians(df['Main_Azimuth']))
    df['cos_Azimuth'] = np.cos(np.radians(df['Azimuth']))

    df['Main_Vector_X'], df['Main_Vector_Y'], df['Main_Vector_Z'] = zip(*df.apply(lambda row: convert_to_vector(row['Main_Longitude'], row['Main_Latitude'], row['Main_Azimuth']), axis=1))
    df['Vector_X'], df['Vector_Y'], df['Vector_Z'] = zip(*df.apply(lambda row: convert_to_vector(row['Longitude'], row['Latitude'], row['Azimuth']), axis=1))

    df.to_csv('processed_dataset.csv', index=False)


def generate_2g_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Iterate through each column to find the one containing the "2G" pattern
    for column in df.columns:
        # Filter rows containing the "2G" pattern
        filtered_df = df[df[column].astype(str).str.contains('2G')]
        
        # If rows containing "2G" pattern are found in the column, write to output CSV
        if not filtered_df.empty:
            # Write the filtered DataFrame to the output CSV file
            filtered_df.to_csv(output_file, index=False)
            print("Output file '{}' generated successfully.".format(output_file))
            return
        
    # If no rows containing "2G" pattern are found in any column
    print("No rows containing the '2G' pattern found in the input CSV file.")

    
def find_replace_file(input_file, output_file):
    # Open the input file and read its content
    with open(input_file, 'r') as f:
        content = f.read()

    # Define the pattern for finding data
    pattern = r'(2G|3G|4G)-\d+'
    
    # Define a function to perform the replacement
    def repl(match):
        return match.group(1)
    
    # Perform the find and replace operation
    modified_content = re.sub(pattern, repl, content)
    
    # Write the modified content to the output file
    with open(output_file, 'w') as f:
        f.write(modified_content)



