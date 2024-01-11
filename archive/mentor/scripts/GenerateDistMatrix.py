import pandas as pd
import requests
import configparser
import numpy as np
import tqdm
import time
import datetime
import sys
import os

def get_matrix_data(coordinates, access_token):
    """
    Fetch travel time matrix using Mapbox Matrix API. 
    Sets the first coordinate as the source and the rest as destinations.

    :param coordinates: List of coordinates [longitude, latitude]
    :param access_token: Your Mapbox Access Token
    :return: JSON response from Mapbox API
    """
    # Convert list of coordinates to string format
    coordinates_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])

    # Endpoint URL (assuming driving mode here, but can be changed to walking, cycling, etc.)
    url = f"https://api.mapbox.com/directions-matrix/v1/mapbox/driving/{coordinates_str}"

    # Parameters
    params = {
        "access_token": access_token,
        "annotations": "distance",
        "sources": "0",
        # Destinations are all coordinates after the first one
        "destinations": ";".join([str(i) for i in range(1, len(coordinates))]),
    }

    # Make the API call
    response = requests.get(url, params=params)

    #check if the response is valid
    if response.status_code != 200:
        raise Exception(f"Error fetching distances from Mapbox API: {response.text}")

    # Return the JSON response
    return response.json()

def read_df(data_dir):
    """
    Initialize the data from a CSV file.

    :param data_dir: Path to the data directory.
    :return: DataFrame with a 'Coordinates' column, combining 'Longitude' and 'Latitude' into a list.
    :raises FileNotFoundError: If the specified file is not found.
    :raises ValueError: If 'Longitude' or 'Latitude' columns are missing.
    """

    # Take file name from the command line
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        raise ValueError("File name must be provided.")
    
    csv_path = os.path.join(data_dir, file_name)

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' was not found.")

    # Check for required columns
    if 'Longitude' not in df.columns or 'Latitude' not in df.columns:
        raise ValueError("The DataFrame must contain 'Longitude' and 'Latitude' columns.")

    # Convert coordinates to list of lists
    df['Coordinates'] = df[['Longitude', 'Latitude']].values.tolist()
    df.drop(columns=['Longitude', 'Latitude'], inplace=True)

    return df


def read_config(root_dir):
    """
    Initialize the config file

    :param root_dir: Path to the root directory.
    :return: Mapbox token
    """
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config_path = os.path.join(root_dir, 'config.ini')    
    config.read(config_path)
    mapbox_token = config['mapbox']['token']

    return mapbox_token


def generated_matrix(df, mapbox_token):
    """
    Generate the distance matrix using the Mapbox API.

    :param df: DataFrame with a 'Coordinates' column
    :param mapbox_token: Mapbox token
    :return: Distance matrix
    """

    # Initialize the matrix
    full_matrix = np.zeros(len(df))

    # Get the matrix data. 
    # Goes through every source once and then every destination for every source.
    col_idx = df.columns.get_loc('Coordinates')
    for i in tqdm.tqdm(range(len(df))):
        horizontal = [[]]
        # Goes through 24 destinations for every source as the mapbox api 
        # is limited to 25 coordinates per call
        for j in range(0, len(df), 24):
            coordinate_list = [df.iloc[i, col_idx]] + df.iloc[j:j+24, col_idx].tolist()
            result = get_matrix_data(coordinate_list, mapbox_token)['distances']
            horizontal = np.hstack((horizontal, result))
            time.sleep(1)
        full_matrix = np.vstack((full_matrix, horizontal))

    # remove the first row which is all zeros
    full_matrix = full_matrix[1:, :]

    return full_matrix


def main():

    # Construct the path to the required directories
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.join(current_dir, '..')
    data_dir = os.path.join(root_dir, 'data')

    # Initialize the data
    df = read_df(data_dir)
    mapbox_token = read_config(root_dir)
    distance_matrix = generated_matrix(df, mapbox_token)

    # Get the current date and time for the file name
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y-%m-%d_%H:%M')
       
    # Save the matrix to a file
    filename = f"{data_dir}/generated_distance_matrices/distance_matrix_{timestamp_str}.npy"
    np.save(filename, distance_matrix)
    print('----Distance Matrix has been generated!----')

if __name__ == '__main__':
    main()