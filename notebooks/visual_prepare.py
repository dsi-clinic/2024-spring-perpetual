#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:31:30 2024

@author: genie_god
"""

import pandas as pd
import glob
import os


def visual_prepare(points, dis, route_dir, output_dir):
    '''
    this function prepares each csv file for visualization

    Parameters
    ----------
    points : string
        master dataset directory
    dis : 2d array
        distance matrix
    route_dir : string
        file directory for all the routes csv file
    output_dir : string
        file directory to store all the outputs *_vic.csv file

    Returns
    -------
    None.

    '''
    gal_points = pd.read_csv(points)
    gal_dis = pd.read_csv(dis)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Pattern to match all CSV files in the directory
    route_files_pattern = route_dir + "route_*.csv"

    # List of columns to keep
    columns_to_keep = ["Name", "Latitude", "Longitude", "Address",
                       "Weekly_Dropoff_Totes", "Daily_Pickup_Totes",
                       "pickup_type"]

    # Iterate over each route CSV file in the directory
    for route_file in glob.glob(route_files_pattern):
        # Load the route CSV file into a DataFrame
        route_df = pd.read_csv(route_file)

        result_df = gal_points.loc[route_df['Node'].values]

        df_filtered = result_df[columns_to_keep]

        output_file_path = os.path.join(output_dir,
                                        os.path.basename(route_file).
                                        replace(".csv", "_vis.csv"))

        df_filtered.to_csv(output_file_path, index=False)


def main():
    gal_points = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_pts_galv.csv")
    gal_dis = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_distances_galv.csv")
    # Path to the directory containing route CSV files
    route_dir = "routes/route_1h_100/"
    output_dir = "route_vis/"
    visual_prepare(gal_points, gal_dis, route_dir, output_dir)


if __name__ == "__main__":
    main()
