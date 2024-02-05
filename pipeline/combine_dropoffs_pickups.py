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

    # set up distance matrix's indices for duplication
    dists_df.columns = [i for i in range(len(dists_df.columns))]

    # duplicate columns + fill in capacity column (combined pickup/dropoffs)
    res_df, res_dists = df, dists_df

    capacities = list(df['Daily_Pickup_Totes'])
    for i, row in df.iterrows():
        if row['Weekly_Dropoff_Totes'] != 0:
            capacities.append(-1 * row['Weekly_Dropoff_Totes'])
            row_copy = pd.DataFrame(row).T
            res_df = pd.concat([res_df, row_copy])
    
            dist_copy = pd.DataFrame(res_dists.iloc[i]).T
            res_dists = pd.concat([res_dists, dist_copy])
    
            res_dists[i+.1] = res_dists[i]
    res_df['Capacity'] = capacities
    res_dists.columns = res_dists.index

    # save outputs
    res_df.to_csv(output_res_df_path)
    res_dists.to_csv(output_res_dist_path)
    