#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 09:31:30 2024

@author: genie_god
"""

import pandas as pd
import glob
import os


def main():
    gal_points = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_pts_galv.csv")
    gal_dis = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_distances_galv.csv")

    # Path to the directory containing route CSV files
    route_dir = "routes/route_1h_100/"
    output_dir = "route_vis/"

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
    
    '''
    for route_file in glob.glob(route_files_pattern):
        route_df = pd.read_csv(route_file)
        # Ensure Node values are integers if they're used as indices
        route_df['Node'] = route_df['Node'].astype(int)

        # Initialize a list to store distances
        distances = []

        # Iterate over the DataFrame rows
        for i in range(len(route_df)):
            cur_node = route_df.iloc[i]['Node']
            if i == 0:
                # For the first node, there's no previous node to calculate distance from
                # You might want to handle this case differently depending on your data
                distances.append(0)
            else:
                pre_node = route_df.iloc[i-1]['Node']
                # Access the distance from the distance matrix
                # Adjust this line if necessary to match your distance matrix access method
                distance = gal_dis.loc[pre_node, cur_node]
                distances.append(distance)

        # Add the distances to the DataFrame
        route_df['Distance'] = distances

        # Filter the DataFrame
        result_df = gal_points.loc[route_df['Node'].values]
        df_filtered = result_df[columns_to_keep]
        # Add the distance column
        df_filtered['Distance'] = route_df['Distance'].values

        # Save to CSV
        output_file_path = os.path.join(output_dir, os.path.basename(
            route_file).replace(".csv", "_vis.csv"))
        df_filtered.to_csv(output_file_path, index=False)

    '''
if __name__ == "__main__":
    main()
