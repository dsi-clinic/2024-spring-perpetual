import pandas as pd
from pipeline.utils.cfgparser import read_cfg
from pipeline.utils.google_cvrp import solve_and_save

if __name__ == '__main__':
    
    cfg = read_cfg("../pipeline/utils/config_inputs.ini",
                   "solve.pickups_and_dropoffs")

    df = pd.read_csv(cfg["combined_df_path"])
    df.at[0, 'Capacity'] = np.sum(df["Weekly_Dropoff_Totes"])
    df.to_csv(cfg["combined_df_path"], index = False)

    solve_and_save(
        path_locations_df=cfg["combined_df_path"],
        path_distance_matrix=cfg["combined_dist_path"],
        num_vehicles=int(cfg["combined_num_vehicles"]),
        vehicle_capacity=int(cfg["vehicle_capacity"]),
        num_seconds=int(cfg["combined_sim_duration"]),
        capacity=cfg["capacity_combined"],
        depot_index=int(cfg["depot_index"]),
        output_path=cfg["combined_route_dir"],
    )