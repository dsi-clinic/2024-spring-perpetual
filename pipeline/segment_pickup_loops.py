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

    
    res_df = pd.DataFrame()
    res_dists = dists
    connected = np.matrix(np.ones((len(df)-1,len(df)-1)) * 1)
    connected = np.append([np.ones(len(df)-1)],connected, 0)
    connected = np.append([[1] for _ in range(len(df))], connected, 1)

    first = True
    for root, _, files in os.walk(route_dir):
        for filename in fnmatch.filter(files, "*.csv"):
            filepath = os.path.join(root, filename)
            if os.path.isfile(filepath):
                
                print(f"segment_pickup_loops :: csv detected: {filename}")
    
                name = filename[:-4]
                route_data = pd.read_csv(filepath)
                if first: # include depot
                    rt_idx = route_data["Original_Index"][:-1]
                    first = False
                else: # exclude depot (will repeat depot if not done)
                    rt_idx = route_data["Original_Index"][1:-1]
                res_df = pd.concat([res_df, df.loc[rt_idx]])
                for i in rt_idx:
                    for j in rt_idx:
                        connected[i,j] = 1
    
    seg_dists = pd.DataFrame(np.multiply(res_dists.to_numpy(), connected))
    seg_dists.columns = [str(i) for i in range(len(df))]
    seg_dists.head()

    res_idx = res_df.index
    seg_dists = seg_dists.loc[res_idx][[str(i) for i in res_idx]]

    res_df.to_csv(cfg["seg_df_path"], index = False)
    seg_dists.to_csv(cfg["seg_dists_path"], index = False)

if __name__ == "__main__":
    main()


