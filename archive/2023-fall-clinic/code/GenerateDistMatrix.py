import ast
import configparser
import datetime
import pickle
import time

import numpy as np
import pandas as pd
import requests
import tqdm


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

    # Endpoint URL (assuming driving mode here,
    # but can be changed to walking, cycling, etc.)
    url_root = "https://api.mapbox.com/directions-matrix/v1/mapbox/driving"
    url = f"{url_root}/{coordinates_str}"

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


# not used in this file
def generate_capacity_list(df, timestamp_str):
    """
    Generate a list of capacities from the dataframe.

    :param df: Dataframe
    :param timestamp_str: Timestamp string
    """
    # save list as pkl file
    capacity_file = f"data/capacity_list_{timestamp_str}.pkl"
    with open(capacity_file, "wb") as f:
        pickle.dump(list(map(int, df["Daily_Pickup_Totes"].tolist())), f)


def add_coordinates(file_name):
    """
     Read a CSV file, capitalize column names,
     and add a new 'Coordinates' column.

    :param file_name: String
    """
    df = pd.read_csv(file_name)
    df.columns = [col.capitalize() for col in df.columns]
    df["Coordinates"] = df[["Longitude", "Latitude"]].values.tolist()
    df.drop(columns=["Longitude", "Latitude"], inplace=True)
    return df


def generate_coordinate_list(df_list):
    """
    Generate a list of coordinates by extracting 'Coordinates' column
    from a list of DataFrames.

    :param df_list: list of pandas.DataFrame
    :return: list
    """
    coordinate_list = []
    for df in df_list:
        for coordinate in df["Coordinates"]:
            coordinate_list.append(coordinate)
    return coordinate_list


def initialize_data():
    """
    Initialize the data from the CSV file
    and the Mapbox token from the config file.

    :return: Dataframe and Mapbox token
    """

    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config.read("../utils/config_mapbox.ini")
    mapbox_token = config["mapbox"]["token"]

    # Take file name from config
    config.read("../utils/config_inputs.ini")
    file_name_indoor = config["original data source"]["indoor"]
    file_name_outdoor = config["original data source"]["outdoor"]

    # Convert coordinates to list of lists
    df_indoor = add_coordinates(file_name_indoor)
    df_outdoor = add_coordinates(file_name_outdoor)

    return df_indoor, df_outdoor, mapbox_token


def generate_distance_matrix():
    # Initialize the data
    df_indoor, df_outdoor, mapbox_token = initialize_data()

    # prepare the coordination list
    pickup_cor_list = generate_coordinate_list([df_indoor, df_outdoor])
    config = configparser.ConfigParser()
    config.read("../utils/config_inputs.ini")
    source_location = ast.literal_eval(config["original data source"]["source"])
    coordinate_list = [source_location] + pickup_cor_list

    # Initialize the matrix
    full_matrix = np.zeros(len(coordinate_list))

    # Get the matrix data.
    # Goes through every source once
    # and then every destination for every source.
    for i in tqdm.tqdm(range(len(coordinate_list))):
        horizontal = [[]]
        # Goes through 24 destinations for every source due to api limit
        for j in range(0, len(coordinate_list), 24):
            c_list = [coordinate_list[i]] + coordinate_list[j : j + 24]
            result = get_matrix_data(c_list, mapbox_token)["distances"]
            horizontal = np.hstack((horizontal, result))
            time.sleep(1)
        full_matrix = np.vstack((full_matrix, horizontal))

    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y%m%d_%H%M%S")

    # remove the first row which is all zeros
    full_matrix = full_matrix[1:, :]

    # Save the matrix to a file
    filename_root = "../data/generated_distance_matrices"
    filename = f"{filename_root}/distance_matrix_{timestamp_str}.npy"
    np.save(filename, full_matrix)

    # Ensure the 'Matrix Dir' section exists
    if "Matrix Dir" not in config:
        config["Matrix Dir"] = {}

    # Assign the filename to the 'distance matrix' key
    # in the 'Matrix Dir' section
    config["Matrix Dir"]["distance_matrix"] = filename

    # Write the configuration to the config_inputs.ini file
    with open("../utils/config_inputs.ini", "w") as configfile:
        config.write(configfile)

    print(f"full distance matrix generated at {filename}")


if __name__ == "__main__":
    generate_distance_matrix()
