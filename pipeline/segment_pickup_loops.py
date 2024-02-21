import ast
import fnmatch
import os

import pandas as pd
import numpy as np

from pipeline.utils.cfg_parser import read_cfg

def main():

     # read config for finding intra-route distances
    cfg = read_cfg(
        "../pipeline/utils/config_inputs.ini", "segment.pickup_loops"
    )
    
    df = pd.read_csv(cfg["truth_df_path"])
    dists = pd.read_csv(cfg["truth_dists_path"])

    route_dir = cfg["pickup_routes_dir"]

    first = True
    for filename in fnmatch.filter(next(os.walk(route_dir))[2], "route*.csv"):
        filepath = os.path.join(route_dir, filename)
        if os.path.isfile(filepath):
            
            print(f"segment_pickup_loops :: csv detected: {filename}")

            name = filename[:-4]
            route_data = pd.read_csv(filepath)
            rt_idx = route_data["Original_Index"][:-1]

            res_df = df.loc[rt_idx]
            df_path = os.path.join(cfg["segmented_dir"], name + "_pts.csv")
            res_df.to_csv(df_path, index = False)

            res_dists = dists.loc[rt_idx][[str(i) for i in rt_idx]]
            dists_path = os.path.join(cfg["segmented_dir"], name + "_dists.csv")
            res_dists.to_csv(dists_path, index = False)

if __name__ == "__main__":
    main()


