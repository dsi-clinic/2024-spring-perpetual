import pandas as pd
import configparser


if __name__ == 'main':

    # read config for combining dropoffs and pickups
    config = configparser.ConfigParser()
    cfg = config.read("../pipeline/utils/config_inputs.ini", "combine.dropoffs_pickups")

    # parse config
    df_path = cfg["toy_example_df_path"]
    dist_path = cfg["toy_example_dist_path"]
    output_res_df_path = cfg["output_res_df_path"]
    output_res_dist_path = cfg["output_res_dist_path"]

    # read input dataframes
    df = pd.read_csv(df_path)
    dists_df = pd.read_csv(dist_path)

    # set up for 
    dists_df.columns = [i for i in range(len(dists_df.columns))]

    res_df, res_dists = df, dists_df

    for i, row in df.iterrows():
        if row['Weekly_Dropoff_Totes'] != 0:
            row_copy = pd.DataFrame(row).T
            row_copy['Daily_Pickup_Totes'] = -1 * row_copy['Weekly_Dropoff_Totes']
            res_df = pd.concat([res_df, row_copy])
    
            dist_copy = pd.DataFrame(res_dists.iloc[i]).T
            res_dists = pd.concat([res_dists, dist_copy]) 
            if res_dists.isnull().values.any():
                print(f"NaN detected while duplicating rows in res_dists on iteration {i}")
                print(res_dists)
                break
            res_dists[i+.1] = res_dists[i]

    
    res_df = res_df.drop("Weekly_Dropoff_Totes", axis = 1).rename(columns={"Daily_Pickup_Totes": "Capacity"})
    res_dists.columns = res_dists.index
        

    res_df.to_csv(output_res_df_path)
    res_dists.to_csv(output_res_dist_path)
    