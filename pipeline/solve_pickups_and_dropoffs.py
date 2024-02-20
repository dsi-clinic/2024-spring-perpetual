import numpy as np
import pandas as pd

from pipeline.utils.cfg_parser import read_cfg
from pipeline.utils.google_cvrp import solve_and_save

def main():
    # read config for combining dropoffs and pickups
    cfg = read_cfg(
        "../pipeline/utils/config_inputs.ini", "solve.pickups_and_dropoffs"
    )

    print(
        f"""solve_pickups_and_dropoffs :: running on
          {cfg["combined_df_path"]} for {cfg["sim_duration"]}s"""
    )
    # solve the intra-route pickup/dropoff problem
    solve_and_save(
        path_locations_df=cfg["combined_df_path"],
        path_distance_matrix=cfg["combined_dists_path"],
        num_vehicles=int(cfg["num_vehicles"]),
        vehicle_capacity=int(cfg["vehicle_capacity"]),
        num_seconds=int(cfg["sim_duration"]),
        capacity=cfg["capacity_col"],
        depot_index=int(cfg["depot_index"]),
        output_path=cfg["combined_routes_dir"],
    )

if __name__ == "__main__":
    main()
