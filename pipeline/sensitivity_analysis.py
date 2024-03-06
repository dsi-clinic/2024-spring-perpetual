import pandas as pd
import numpy as np

import fnmatch
import json
import os

from pipeline.utils.filter_df import filter_df

def sensitivity_analysis(route_dir):
    '''
    Given a directory with a single csv with all routes,
    summarize number of routes, pickup/dropoff demands, and
    distance traveled.

    inputs:
        'route_dir' : str
            Path to directory with route csv in it
    '''
    
    route_path = route_dir + "routes.csv"
    routes = pd.read_csv(route_path)

    # set up storage dictionary for return DataFrame/csv
    res_dict = {
                'Route_Names':[],
                'Pickup_Demands':[],
                'Pickup_Estimated_Cups':[],
                'Dropoff_Demands':[],
                'Dropoff_Estimated_Cups':[],
                'Cumulative_Distances':[]
    }

    # for each route, fill dictionary with summary information and estimates
    total_pickup_tote = 0
    total_pickup_cups = 0
    total_dropoff_totes = 0
    total_dropoff_cups = 0
    total_dist_trav = 0
    for assignment in routes["Route"].unique():
    
        df = filter_df(routes, "Route", assignment)
        
        res_dict['Route_Names'].append(assignment)
        pickup_demand_total = np.sum(df['Daily_Pickup_Totes'])
        res_dict['Pickup_Demands'].append(pickup_demand_total)
        res_dict['Pickup_Estimated_Cups'].append(50*pickup_demand_total)
        dropoff_demand_total = np.sum(df['Weekly_Dropoff_Totes'])
        res_dict['Dropoff_Demands'].append(dropoff_demand_total)
        res_dict['Dropoff_Estimated_Cups'].append(250*pickup_demand_total)
        distance_total = df.iloc[-1]['Cumulative_Distance']
        res_dict['Cumulative_Distances'].append(distance_total)

        total_pickup_totes += pickup_demand_total
        total_pickup_cups += 50*pickup_demand_total
        total_dropoff_totes += dropoff_demand_total
        total_dropoff_cups += 250*dropoff_demand_total
        total_dist_trav += distance_total

    # save totals
    res_dict['Route_Names'].append('Totals')
    res_dict['Pickup_Demands'].append(total_pickup_totes)
    res_dict['Pickup_Estimated_Cups'].append(total_pickup_cups)
    res_dict['Dropoff_Demands'].append(total_dropoff_totes)
    res_dict['Dropoff_Estimated_Cups'].append(total_dropoff_cups)
    res_dict['Cumulative_Distances'].append(total_dist_trav)
    
    # save dataframe
    pd.DataFrame(res_dict).to_csv(route_dir + "sensitivity_analysis.csv")
    


if __name__ == '__main__':
    res_dict = sensitivity_analysis("../data/output/combined_pickups_and_dropoffs/20240306070855/")
    