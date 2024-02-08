import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation
from pipeline.utils.utils import read_cfg

if __name__ == '__main__':

    # read config for combining dropoffs and pickups
    cfg = read_cfg("../pipeline/utils/config_inputs.ini",
                   "combine.dropoffs_pickups")

    # parse config
    pickups_df_path = cfg["pickups_df_path"]
    pickups_dist_path = cfg["pickups_dist_path"]
    combined_df_path = cfg["combined_df_path"]
    combined_dist_path = cfg["combined_dist_path"]

    print(f"combine_dropoffs_pickups :: working on {pickups_df_path}")

    # read input dataframes
    df = pd.read_csv(pickups_df_path)
    dists_df = pd.read_csv(pickups_dist_path)

    res_df, res_dists = df, dists_df
    res_df['Capacity'] = res_df['Daily_Pickup_Totes'] - res_df['Weekly_Dropoff_Totes']

    # save outputs
    res_df.to_csv(combined_df_path, index = False)
    res_dists.to_csv(combined_dist_path, index = False)
    print("combine_dropoffs_pickups :: done!")