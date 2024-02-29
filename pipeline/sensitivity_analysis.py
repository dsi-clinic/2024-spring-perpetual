import pandas as pd
import numpy as np

import fnmatch
import json
import os

def sensitivity_analysis(config):
    
    cfg = config["sensitivity_analysis"]
    route_path = cfg["route_path"] # "../data/outputs/routes/pickups_only/"
    num_routes = 0
    res_dict = {'Num_Vehicles' : cfg["???"],
                'Vehicle_Capacity' : cfg["???"],
                'Route_Names':[],
                'Pickup_Demands':[],
                'Dropoff_Demands':[],
                'Cumulative_Distances':[]}
    for filename in fnmatch.filter(next(os.walk(route_path))[2], "route*.csv"):
        num_routes += 1
        filepath = os.path.join(route_path, filename)
        if os.path.isfile(filepath):
            print(
                f"""solve_pickups_and_dropoffs :: detected {filename}"""
            )
            name = filename[:-4]
            df = pd.read_csv(filepath)
            # last_line = pd.DataFrame(df.iloc[-1]).T
            res_dict['Route_Names'].append(name)
            res_dict['Pickup_Demands'].append(str(np.sum(df['Daily_Pickup_Totes'])))
            res_dict['Dropoff_Demands'].append(str(np.sum(df['Weekly_Dropoff_Totes'])))
            res_dict['Cumulative_Distances'].append(str(df.iloc[-1]['Cumulative_Distance']))