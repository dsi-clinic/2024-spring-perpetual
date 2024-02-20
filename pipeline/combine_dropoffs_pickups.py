import pandas as pd
from pipeline.utils.cfg_parser import read_cfg

if __name__ == "__main__":

    # read config for combining dropoffs and pickups
    cfg = read_cfg(
        "../pipeline/utils/config_inputs.ini", "combine.pickups_and_dropoffs"
    )

    # parse config
    segmented_df_path = cfg["segmented_df_path"]
    segmented_dists_path = cfg["segmented_dists_path"]
    combined_df_path = cfg["combined_df_path"]
    combined_dists_path = cfg["combined_dists_path"]

    print(f"combine_dropoffs_pickups :: working on {segmented_df_path}")

    # read input dataframes
    df = pd.read_csv(segmented_df_path)
    dists_df = pd.read_csv(segmented_dists_path)

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
    res_df.to_csv(combined_df_path, index=False)
    res_dists.to_csv(combined_dists_path, index=False)
    print("combine_dropoffs_pickups :: done!")
