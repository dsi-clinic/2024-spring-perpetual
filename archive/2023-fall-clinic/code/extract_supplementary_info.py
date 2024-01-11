# produce supplementary info about point (used for the map popups)

import configparser
import csv

import pandas as pd

if __name__ == "__main__":
    # read config
    config = configparser.ConfigParser()
    config.read("../utils/config_inputs.ini")
    cfg = config["extract.supp_info"]

    # parse config
    truth_df_path = cfg["truth_df_path"]
    supp_info_df_savepath = cfg["supp_info_df_savepath"]

    # read single-truth dataframe
    truth_df = pd.read_csv(truth_df_path)

    # prepare supp_info dictionary
    supp_info = {}
    name_col = "Name"
    supp_cols = [
        "Daily_Pickup_Totes",
        "Weekly_Dropoff_Totes",
        "category",
        "Address",
        "location_type",
        "pickup_type",
    ]
    for i in range(len(truth_df)):
        supp_info[truth_df[name_col].iloc[i]] = truth_df[supp_cols].iloc[i]

    with open("../output/data/extracted_supp_info_dict.csv", "w") as f:
        w = csv.DictWriter(f, supp_info.keys())
        w.writeheader()
        w.writerow(supp_info)
