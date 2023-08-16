import pandas as pd
import requests
import configparser
import numpy as np
import tqdm
import time
import datetime
import sys

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
    # Return the JSON response
    return response.json()


def initialize_data():
    """
    Initialize the data from the CSV file and the Mapbox token from the config file.

    :return: Dataframe and Mapbox token
    """

    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config.read('config.ini')
    mapbox_token = config['mapbox']['token']

    # Take file name from terminal
    file_name = sys.argv[1]
    df = pd.read_csv(file_name, index_col=0)
    df['coordinates'] = df.apply(lambda row: [row['longitude'], row['latitude']], axis=1)
    df = df.drop(columns=['longitude', 'latitude'])

    return df, mapbox_token


def main():

    # Initialize the data
    df, mapbox_token = initialize_data()

    # Initialize the matrix
    full_matrix = np.zeros(len(df))

    # Get the matrix data. Goes through every source once and then every destination for every source.
    for i in tqdm.tqdm(range(len(df))):
        horizontal = [[]]
        # Goes through 24 destinations for every source due to api limit
        for j in range(0, len(df), 24):
            coordinate_list = [df.iloc[i, 1]] + df.iloc[j:j+24, 1].tolist()
            result = get_matrix_data(coordinate_list, mapbox_token)['distances']
            horizontal = np.hstack((horizontal, result))
            time.sleep(1)
        full_matrix = np.vstack((full_matrix, horizontal))

    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y%m%d_%H%M%S')
    # remove the source row
    full_matrix = full_matrix[1:, :]
    # Save the matrix to a file
    filename = f"data/distance_matrix_{timestamp_str}.npy"
    np.save(filename, full_matrix)

    print('Complete!')

if __name__ == '__main__':
    main()