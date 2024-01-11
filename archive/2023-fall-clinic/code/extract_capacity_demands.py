# produce capacity list from the single-truth dataframe (used for routing)

import configparser
import csv

import pandas as pd

if __name__ == "__main__":
    # read config
    config = configparser.ConfigParser()
    config.read("../utils/config_inputs.ini")
    cfg = config["extract.capacity_demands"]

    # parse config
    truth_df_path = cfg["truth_df_path"]
    capacity_df_savepath = cfg["capacity_df_savepath"]

    # read single-truth dataframe
    truth_df = pd.read_csv(truth_df_path)

    # prepare capacity dataframe
    capacities = {}
    name_col = "Name"
    demand_cols = ["Daily_Pickup_Totes", "Weekly_Dropoff_Totes"]

    # prepare capacity dictionary
    for i in range(len(truth_df)):
        capacities[truth_df[name_col].iloc[i]] = truth_df[demand_cols].iloc[i]

    with open("../output/data/extracted_capacities_dict.csv", "w") as f:
        w = csv.DictWriter(f, capacities.keys())
        w.writeheader()
        w.writerow(capacities)
