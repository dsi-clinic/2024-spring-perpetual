import fnmatch
import os
import pandas as pd
from pipeline.utils.cfg_parser import read_cfg

def combine_dropoffs_pickups():

    # read config for combining dropoffs and pickups
    cfg = read_cfg(
        "../pipeline/utils/config_inputs.ini", "combine.pickups_and_dropoffs"
    )

    # parse config
    # segmented_df_path = cfg["segmented_df_path"]
    # segmented_dists_path = cfg["segmented_dists_path"]
    # combined_df_path = cfg["combined_df_path"]
    # combined_dists_path = cfg["combined_dists_path"]
    seg_dir = cfg["segmented_dir"]
    comb_dir = cfg["combined_dir"]

    
    for filename in fnmatch.filter(next(os.walk(seg_dir))[2], "route*_pts.csv"):
        filepath = os.path.join(seg_dir, filename)
        if os.path.isfile(filepath):
            
            print(f"segment_pickup_loops :: csv detected: {filename}")

            name = filename[:-8]

            # read input dataframes
            print(f"combine_dropoffs_pickups :: working on {seg_dir}")
            df = pd.read_csv(filepath)
            dists_df = pd.read_csv(os.path.join(seg_dir, name + "_dists.csv"))

            # set up res dataframes for output
            res_df, res_dists = df, dists_df
            res_dists.columns = [i for i in range(len(res_dists.columns))]

            # create new "Capacity" column (<= 0 for dropoff, >= 0 for pickup)
            dropoff_total = 0
            capacities = list(df["Daily_Pickup_Totes"])
            for i, row in df.iterrows():
                if row["Weekly_Dropoff_Totes"] != 0:
                    dropoff_total += row["Weekly_Dropoff_Totes"]
                    capacities.append(-1 * row["Weekly_Dropoff_Totes"])
                    row_copy = pd.DataFrame(row).T
                    res_df = pd.concat([res_df, row_copy])

                    dists_copy = pd.DataFrame(res_dists.iloc[i]).T
                    res_dists = pd.concat([res_dists, dists_copy])

                    res_dists = pd.concat([res_dists, res_dists[i]], axis = 1)
            capacities[0] = dropoff_total
            res_df["Capacity"] = capacities
            res_dists = res_dists.reset_index(drop=True)
            res_dists.columns = [str(i) for i in range(len(res_dists.columns))]
            
            # save outputs
            res_df.to_csv(os.path.join(comb_dir, filename), index=False)
            res_dists.to_csv(os.path.join(comb_dir, name + "_dists.csv"), index=False)
    print("combine_dropoffs_pickups :: done!")

if __name__ == "__main__":
    combine_dropoffs_pickups()