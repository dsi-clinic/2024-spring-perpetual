"""
complete_feasibility_report.py

This script interacts with the utils/config.ini file to
grab inputs and outputs from the pipeline and update
the feasibility report in place for each new trial

Author: Sarah Walker

1. Run this script in the terminal using:
cd code
python complete_feasibility_report.py
"""

import configparser

import pandas as pd


def extract_route_info(path_to_routes, num_routes):
    """
    Use this function to open each route from the output
    to extract total loads, locations, and distances for each route

    inputs:
    outputs: list of lists - [list of load_per_vehicle,
                            list of locations_per_vehicle,
                            list of distance_per_vehicle]
    """
    load_list = []
    num_loc_list = []
    distance_list = []
    for i in range(num_routes):
        route = pd.read_csv(
            f"{path_to_routes}/route{i + 1}.csv", encoding="unicode_escape"
        )
        load_list.append(route.loc[len(route) - 1, "Truck_Load"])
        num_loc_list.append(len(route) - 2)
        distance_list.append(route.loc[len(route) - 1, "Cumulative_Distance"])

    return load_list, num_loc_list, distance_list


def build_trial_dict(feasibility_report):
    """
    Build a dictionary so every key matches the order of the
    columns of the feasibility file

    input: df - feasibility_report
    output: dict - trial_dict
    """
    trial_dict = {}
    list_of_column_names = feasibility_report.columns.to_list()
    for name in list_of_column_names:
        trial_dict[name] = ""
    return trial_dict


def main():

    trial_dict = build_trial_dict(feasibility_report)

    # create arguments to the extract_route_info function
    path_to_routes = config["optimize google cvrp"]["output_path"]
    num_routes = int(config["optimize google cvrp"]["num_vehicles"])

    # use these arguments to call the function extract_route_info
    (
        load_per_vehicle,
        locations_per_vehicle,
        distance_per_vehicle,
    ) = extract_route_info(path_to_routes, num_routes)

    # assign values to appropriate key of the trial dictionary
    trial_dict["output_path"] = path_to_routes
    trial_dict["num_vehicles"] = num_routes

    trial_dict["load_per_vehicle"] = load_per_vehicle
    trial_dict["locations_per_vehicle"] = locations_per_vehicle
    trial_dict["distance_per_vehicle"] = distance_per_vehicle

    trial_dict["cumulative_load"] = sum(load_per_vehicle)
    trial_dict["total_num_locations"] = sum(locations_per_vehicle)
    trial_dict["cumulative_distance"] = sum(distance_per_vehicle)

    # assign values to remaining keys of the trial dictionary
    # by grabbing the arguments from the config file
    trial_dict["trial_name"] = config["feasibility report"]["trial_name"]
    trial_dict["description"] = config["feasibility report"]["description"]
    trial_dict["path_to_distance_matrix"] = config["optimize google cvrp"][
        "path_to_distance_matrix"
    ]
    trial_dict["path_to_dataframe"] = config["optimize google cvrp"][
        "path_to_dataframe"
    ]
    trial_dict["simulation_run_time"] = int(
        config["optimize google cvrp"]["num_seconds_simulation"]
    )
    trial_dict["vehicle_type"] = config["feasibility report"]["vehicle_type"]
    trial_dict["vehicle_capacity"] = int(
        config["optimize google cvrp"]["vehicle_capacity"]
    )

    # Insert time and cost
    # #trial_dict['time_per_vehicle']
    # #trial_dict['cost_per_vehicle']
    # #trial_dict['cumulative_time']
    # #trial_dict['cumulative_cost']

    trial_dict["path_to_visualizations"] = config["feasibility report"][
        "visualization_path"
    ]
    # trial_dict['name_of_visualizations']

    trial_dict["capacity_type"] = config["optimize google cvrp"]["capacity"]

    # save the trial dictionary as a new row in the feasibility file
    feasibility_report.append(trial_dict, ignore_index=True)
    # save the feasibility file in place
    feasibility_report.to_csv("../output/feasibilityfile.csv")


if __name__ == "__main__":

    # create the config object that will interact with the inputs and outputs
    config = configparser.ConfigParser()
    config.read("../utils/config_inputs.ini")

    # open the feasibility file so you can make edits
    feasibility_report = pd.read_csv("../output/feasibilityfile.csv")

    main()
