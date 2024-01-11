import ast

import folium
import numpy as np
import pandas as pd
from RouteViz import add_markers

from utils.utils import filter_df, filter_dists, read_config

if __name__ == "__main__":

    """
    Table of Contents:
    1. Read config file
    2. Parse config
    3. Read single-truth dataframe and distance matrix
    4. Get distances between bike-served and truck-served locations
    5. Mark truck-served points within {distance_thresh} of bike-served points
    6. Convert marked truck-serviced locations to bike-servicable locations
    7. Aggregate bike capacities at aggregation points
    8. Prepare distance matrices for output
    9. Export everything
    10. Visualize
    """

    # 1. read config
    cfg = read_config("../utils/config_inputs.ini", "convert.to_bikes")

    # 2. parse config
    truth_df_path = cfg["truth_df_path"]
    truth_dist_df_path = cfg["truth_dist_df_path"]
    location = [float(cfg["latitude"]), float(cfg["longitude"])]
    distance_thresh = float(cfg["distance_thresh"])  # in meters
    aggs = ast.literal_eval(cfg["bike_aggregate_list"])

    bike_df_savepath = cfg["bike_df_savepath"]
    truck_pickup_df_savepath = cfg["truck_pickup_df_savepath"]
    truck_dropoff_df_savepath = cfg["truck_dropoff_df_savepath"]
    bike_dist_df_savepath = cfg["bike_dist_df_savepath"]
    truck_pickup_dist_df_savepath = cfg["truck_pickup_dist_df_savepath"]
    truck_dropoff_dist_df_savepath = cfg["truck_dropoff_dist_df_savepath"]

    bike_aggregates_savedir = cfg["bike_aggregates_savedir"]
    map_all_save_path = cfg["map_all_save_path"]
    map_bikes_save_path = cfg["map_bikes_save_path"]
    map_trucks_save_path = cfg["map_trucks_save_path"]

    # 3.0 read single-truth dataframe and single-truth distance matrix
    truth_df = pd.read_csv(truth_df_path)
    truth_dist = pd.read_csv(truth_dist_df_path)
    # 3.1 set up output dataframe (to-be-modified)
    converted_truth_df = truth_df.copy()
    for i in aggs:
        converted_truth_df.at[i, "pickup_type"] = "Bike_Aggregate"

    # 4. get distances between bike-served and truck-served locations
    bike_ind = filter_df(
        converted_truth_df, "pickup_type", "Bike"
    ).index.to_list()
    truck_ind = filter_df(
        converted_truth_df, "pickup_type", "Truck"
    ).index.to_list()
    bike_to_truck_dists = filter_dists(truth_dist, bike_ind, truck_ind)

    # 5. mark truck-served points within {distance_thresh} of bike-served points
    converts_ind = []
    for col in bike_to_truck_dists.columns:
        if min(bike_to_truck_dists[col]) <= distance_thresh:
            converts_ind.append(int(col))

    # 6. convert marked truck-serviced locations to bike-servicable locations
    for i in converts_ind:
        if i not in aggs:
            converted_truth_df.at[i, "pickup_type"] = "Bike_Converted"

    # 7.0. collecting indices for aggregating (7.1.) and outputs (8.)
    total_ind = {i for i in range(len(truth_df))}
    bike_no_agg_ind = sorted(bike_ind + converts_ind)
    bike_all_ind = sorted(bike_no_agg_ind + aggs)
    truck_all_ind = sorted(list(total_ind - set(bike_no_agg_ind)))

    # 7.1. assign aggregate locations for each bike point
    agg_to_bike_dists = filter_dists(truth_dist, aggs, bike_no_agg_ind)
    agg_assignments = {}
    for col in agg_to_bike_dists.columns:
        assignment = int(aggs[np.argmin(agg_to_bike_dists[col])])
        key = bike_no_agg_ind[int(col)]
        converted_truth_df.at[key, "Bike Aggregation Point"] = assignment
        converted_truth_df.at[
            assignment, "Daily_Pickup_Totes"
        ] += converted_truth_df.at[key, "Daily_Pickup_Totes"]

    # 8. prepare info dataframes for output
    truck_converted_df = converted_truth_df[
        (converted_truth_df["pickup_type"] == "Truck")
        | (converted_truth_df["pickup_type"] == "Bike_Aggregate")
    ]
    bike_converted_df = converted_truth_df[
        (converted_truth_df["pickup_type"] == "Bike")
        | (converted_truth_df["pickup_type"] == "Bike_Aggregate")
        | (converted_truth_df["pickup_type"] == "Bike_Converted")
    ]
    # 8.1. remove outdoor points for truck dropoffs
    truck_converted_dropoffs_df = filter_df(
        truck_converted_df, "location_type", "indoor"
    )
    truck_dropoff_ind = truck_converted_dropoffs_df.index.to_list()
    # 8.2. remove bike aggregate point's pickup demand for bikes
    bike_converted_df.loc[
        bike_converted_df.loc[:, "pickup_type"] == "Bike_Aggregate",
        "Daily_Pickup_Totes",
    ] = 0
    # 8.3. prepare distance matrices for output
    truck_dist_df = filter_dists(truth_dist, truck_all_ind, truck_all_ind)
    bike_dist_df = filter_dists(truth_dist, bike_all_ind, bike_all_ind)
    truck_dropoff_dist_df = filter_dists(
        truth_dist, truck_dropoff_ind, truck_dropoff_ind
    )

    # 9. export dataframes
    truck_converted_df.to_csv(truck_pickup_df_savepath)
    truck_converted_dropoffs_df.to_csv(truck_dropoff_df_savepath)
    bike_converted_df.to_csv(bike_df_savepath)
    truck_dist_df.to_csv(truck_pickup_dist_df_savepath, index=False)
    truck_dropoff_dist_df.to_csv(truck_dropoff_dist_df_savepath, index=False)
    bike_dist_df.to_csv(bike_dist_df_savepath, index=False)

    # 10. visualize
    trucks = converted_truth_df[converted_truth_df["pickup_type"] == "Truck"]
    bikes = converted_truth_df[converted_truth_df["pickup_type"] == "Bike"]
    bike_convs = converted_truth_df[
        converted_truth_df["pickup_type"] == "Bike_Converted"
    ]
    bike_aggs = converted_truth_df[
        converted_truth_df["pickup_type"] == "Bike_Aggregate"
    ]

    map_all = folium.Map(
        location=location, tiles="OpenStreetMap", zoom_start=11
    )
    add_markers(map_all, trucks, "blue")
    add_markers(map_all, bikes, "red")
    add_markers(map_all, bike_convs, "green")
    add_markers(map_all, bike_aggs, "orange")
    map_all.save(map_all_save_path)

    map_truck = folium.Map(
        location=location, tiles="OpenStreetMap", zoom_start=11
    )
    add_markers(map_truck, trucks, "blue")
    add_markers(map_truck, bike_aggs, "orange")
    map_truck.save(map_trucks_save_path)

    map_bikes = folium.Map(
        location=location, tiles="OpenStreetMap", zoom_start=11
    )
    add_markers(map_bikes, bikes, "red")
    add_markers(map_bikes, bike_convs, "green")
    add_markers(map_bikes, bike_aggs, "orange")
    map_bikes.save(map_bikes_save_path)
