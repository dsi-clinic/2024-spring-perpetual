#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:43:52 2024

@author: genie_god
"""

"""
after parsing the result from gurobi output, these functions will help to add
more details to each route for the last step parameter analysis
"""

import pandas as pd
import glob
import os
def visual_prepare(points, dis, route_dir, output_dir):
    '''
       this function add more information for each loaction, by grabbing
       informations from master dataset of participation locations

       Parameters
       ----------
       points : string
           directory for dataset of all the points
       dis : string
           directory for distance matrix for each pair of location
       route_dir : string
           directory for all the routes dataframe
       output_dir : string
           directory for outputs

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

        concat_df_route = pd.concat([result_df, df_filtered], axis=1)

        # Create a new column with the index values
        concat_df_route['Node'] = concat_df_route.index
        concat_df_route.reset_index(drop=True, inplace=True)
        add_accumulate_value(concat_df_route, gal_dis)
        output_file_path = os.path.join(output_dir,
                                        os.path.basename(route_file).
                                        replace(".csv", "_vis.csv"))

        concat_df_route.to_csv(output_file_path)


def add_accumulate_value(df_route, dis):
    '''
    add following column to route_* csv file
    df_route['accumulate_pickup']
    df_route['accumulate_dropoff']
    df_route['accumulate_distance']

    Parameters
    ----------
    df_route : dataframe

    Returns
    -------
    updated dataframe with accumulated features

    '''

    number_of_rows = len(df_route)

    accumulated_distance = [0]

    for row in range(1, number_of_rows):
        n = df_route['Node'].iloc[row-1]
        m = df_route['Node'].iloc[row]
        distance = dis.iloc[n, m]
        accumulate_dis = accumulated_distance[-1] + distance
        accumulated_distance.append(accumulate_dis)

    df_route['Cumulative_Distance'] = accumulated_distance

    accumulate_col(df_route, 'Daily_Pickup_Totes',
                   number_of_rows, 'Truck_Pickup_Load')

    accumulate_col(df_route, 'Weekly_Dropoff_Totes',
                   number_of_rows, 'Truck_Dropoff_Load')


def accumulate_col(df, col_name, nrow, accu_col):
    '''
    utility function to calculate accumulated value for the colu_name

    Parameters
    ----------
    df : dataframe
    col_name : string
        column that need to be calculated
    nrow : int
        number of rows in the dataset
    accu_col : string
        column that stores the accumulate value

    Returns
    -------
    None.

    '''
    start = df[col_name].iloc[0][0]
    accumulate = [start]
    for row in range(1, nrow):
        value = df[col_name].iloc[row][0]
        accumulate_row = accumulate[-1] + value
        accumulate.append(accumulate_row)

    df[accu_col] = accumulate
