import configparser
import os
import time

import pandas as pd
import requests
import tqdm


def get_list_data(coordinates, access_token):
    """
    Fetch travel time list using Mapbox Direction API.
    :param coordinates: List of coordinates [longitude, latitude]
    :param access_token: Your Mapbox Access Token
    :return: JSON response from Mapbox API
    """
    # Convert list of coordinates to string format
    coordinates_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])

    # Endpoint URL (assuming driving mode here,
    # but can be changed to walking, cycling, etc.)
    url_root = "https://api.mapbox.com/directions/v5/mapbox/driving"
    url = f"{url_root}/{coordinates_str}?"

    # Parameters
    params = {
        "access_token": access_token,
        "annotations": "duration",
    }

    # Make the API call
    response = requests.get(url, params=params)
    # Return the JSON response
    return response.json()


def add_coordinator(df):
    df["Coordinates"] = df[["Longitude", "Latitude"]].values.tolist()
    df.drop(columns=["Longitude", "Latitude"], inplace=True)
    return df


def initialize_data(route_path):
    """
    Initialize the data from the CSV file

    :return: Dataframe
    """
    file_name = route_path
    df = pd.read_csv(file_name)
    # Convert coordinates to list of lists
    df = add_coordinator(df)

    return df


def get_token():
    """
    Initialize the token from config file

    :return: token
    """
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config.read("config.ini")
    mapbox_token = config["mapbox"]["token"]

    return mapbox_token


def request_time(df, mapbox_token):
    """
    requesting duration for each stop, add 5 min(300 seconds) at each
    location

    """
    # first stop, initialize 5 min loading time
    col_duration = [300]
    total_time = 300
    col_idx = df.columns.get_loc("Coordinates")

    for i in tqdm.tqdm(range(0, len(df) - 1, 24)):
        # Goes through 24 destinations for every source due to api limit
        coordinate_list = df.iloc[i : i + 25, col_idx].tolist()
        result = get_list_data(coordinate_list, mapbox_token)
        # pull out duration list from result
        leg_list = result["routes"][0]["legs"]
        time_list = [i["duration"] for i in leg_list]
        # accumulate time along the route
        for j in range(len(time_list) - 1):
            total_time += time_list[j] + 5 * 60
            col_duration.append(int(total_time))
        time.sleep(1)

    return col_duration


def get_file_name(file_path):
    # Split by '/' and get the last part (the filename)
    file_name_with_extension = file_path.split("/")[-1]

    # Split by '.' and get the first part
    file_name_string = file_name_with_extension.split(".")[0]

    return file_name_string


def add_time_to_route():
    # Initialize the data and tokein
    mapbox_token = get_token()

    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config.read("../utils/config_inputs.ini")

    # Iterate over each route in the config file
    for route_ in config["routes"]:
        file_path = config["routes"][route_]

        # Initialize data for the current route
        df = initialize_data(file_path)

        # Prequsting duration for the current route
        df["duration"] = request_time(df, mapbox_token)

        # Save the DataFrame to a CSV file under data file
        # Extract the base name without the .csv extension
        base_name = os.path.splitext(file_path)[0]

        output_file = base_name + "_time.csv"

        df.to_csv(output_file, index=False)

        print(
            f"Processed route {route_}, "
            "output duration saved to {output_file}"
        )

        # Ensure the 'Matrix Dir' section exists
        if "route_time" not in config:
            config["route_time"] = {}

        file_name_string = get_file_name(output_file)

        # Assign the filename to the route_?_way_points
        config["route_time"][f"{file_name_string}"] = output_file

        # Write the configuration to an INI file
        with open("../utils/config_inputs.ini", "w") as configfile:
            config.write(configfile)

    print("\nComplete adding duration to routes!")


if __name__ == "__main__":
    add_time_to_route()
