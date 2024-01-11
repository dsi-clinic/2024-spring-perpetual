#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:36:59 2023

@author: genie_god
"""

import configparser
import os
import time

import pandas as pd
import requests
import tqdm


def get_route_data(coordinates, access_token):
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
        # "annotation": "duration",
        "steps": "true",
        "waypoints_per_route": "true",
    }

    # Make the API call
    response = requests.get(url, params=params)
    # Return the JSON response
    return response.json()


def add_coordinator(df):
    """
    add coordinator coloumn to the dataframe, which will use for later api
    requesting and visualization

    """
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


def request_waypoints(df, mapbox_token):
    waypoints_location = []
    waypoints_instruction = []
    location_index = []
    way_points_index = []
    col_idx = df.columns.get_loc("Coordinates")

    for i in tqdm.tqdm(range(0, len(df), 24)):
        # Goes through 24 destinations for every source due to api limit
        coordinate_list = df.iloc[i : i + 24, col_idx].tolist()
        result = get_route_data(coordinate_list, mapbox_token)
        # pull out duration list from result
        legs = result["routes"][0]["legs"]
        for j in range(len(legs)):
            steps = legs[j]["steps"]
            for k in range(len(steps)):
                waypoints_location.append(
                    legs[j]["steps"][k]["maneuver"]["location"]
                )
                waypoints_instruction.append(
                    legs[j]["steps"][k]["maneuver"]["instruction"]
                )
                location_index.append(i + j)
                way_points_index.append(k)

        time.sleep(1)

    df_waypoints = pd.DataFrame(
        {
            "location": location_index,
            "way_points": way_points_index,
            "waypoints_location": waypoints_location,
            "waypoints_instruction": waypoints_instruction,
        }
    )
    return df_waypoints


def add_name_address_to_waypoints(df, df_waypoints):
    # Initialize new columns with default values
    df_waypoints["name"] = None
    df_waypoints["address"] = None

    # Iterate over the df_waypoints DataFrame
    for index, row in df_waypoints.iterrows():
        if row["way_points"] == 0:

            df_waypoints.at[index, "name"] = df.at[row["location"], "Name"]
            df_waypoints.at[index, "address"] = df.at[
                row["location"], "Address"
            ]

    return df_waypoints


def get_file_name(file_path):
    # Split by '/' and get the last part (the filename)
    file_name_with_extension = file_path.split("/")[-1]

    # Split by '.' and get the first part
    file_name_string = file_name_with_extension.split(".")[0]

    return file_name_string


def add_way_points_to_route():
    # Initialize the data and token
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

        # Process waypoints for the current route
        df_waypoints = request_waypoints(df, mapbox_token)
        df_waypoints = add_name_address_to_waypoints(df, df_waypoints)

        # Save the DataFrame to a CSV file under data file

        # Extract the base name without the .csv extension
        base_name = os.path.splitext(file_path)[0]

        output_file = base_name + "_waypoints.csv"

        df_waypoints.to_csv(output_file, index=False)

        print(f"Processed route {route_}, output saved to {output_file}")

        # Ensure the 'Matrix Dir' section exists
        if "way_points" not in config:
            config["way_points"] = {}

        file_name_string = get_file_name(output_file)

        # Assign the filename to the route_?_way_points
        config["way_points"][f"{file_name_string}"] = output_file

        # Write the configuration to an INI file
        with open("../utils/config_inputs.ini", "w") as configfile:
            config.write(configfile)

    print("\nComplete adding waypoints to routes!")


if __name__ == "__main__":
    add_way_points_to_route()
