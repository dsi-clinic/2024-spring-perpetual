import pandas as pd
from pipeline.utils.cfgparser import read_cfg

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

    # create new "Capacity" column = "Pickups" - "Dropoffs"
    res_df, res_dists = df, dists_df
    res_dists.columns = [i for i in range(len(res_dists.columns))]

    capacities = list(df['Daily_Pickup_Totes'])
    for i, row in df.iterrows():
        if row['Weekly_Dropoff_Totes'] != 0:
            capacities.append(-1 * row['Weekly_Dropoff_Totes'])
            row_copy = pd.DataFrame(row).T
            res_df = pd.concat([res_df, row_copy])

            dist_copy = pd.DataFrame(res_dists.iloc[i]).T
            res_dists = pd.concat([res_dists, dist_copy])

            res_dists[len(res_dists.columns)] = res_dists[i]
    res_df['Capacity'] = capacities
    res_dists = res_dists.reset_index(drop = True)

    # save outputs
    res_df.to_csv(combined_df_path, index = False)
    res_dists.to_csv(combined_dist_path, index = False)
    print("combine_dropoffs_pickups :: done!")