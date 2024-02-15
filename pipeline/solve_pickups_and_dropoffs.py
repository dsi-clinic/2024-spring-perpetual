import numpy as np
import pandas as pd

from pipeline.utils.cfg_parser import read_cfg
from pipeline.utils.google_cvrp import solve_and_save

if __name__ == "__main__":

    # read config for combining dropoffs and pickups
    cfg = read_cfg(
        "../pipeline/utils/config_inputs.ini", "solve.pickups_and_dropoffs"
    )

    # read df to symbolically fill truck up with clean bins for dropoff
    df = pd.read_csv(cfg["combined_df_path"])
    df.at[0, "Capacity"] = -1 * np.sum(df[df["Capacity"] < 0]["Capacity"])
    df.to_csv(cfg["combined_df_path"], index=False)

    print(
        f"""solve_pickups_and_dropoffs :: running on
          {cfg["combined_df_path"]} for {cfg["combined_sim_duration"]}s"""
    )
    # solve the intra-route pickup/dropoff problem
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
