import fnmatch
import os

from pipeline.utils.cfg_parser import read_ini
from pipeline.utils.google_cvrp import solve_and_save


def solve_pickups_and_dropoffs():

    # read config for combining dropoffs and pickups
    cfg = read_ini(
        "../pipeline/utils/config_inputs.ini", "solve.pickups_and_dropoffs"
    )

    # parse config
    comb_dir = cfg["combined_dir"]
    comb_routes_dir = cfg["combined_routes_dir"]

    # solve the intra-route pickup/dropoff problem
    num_preceding_routes = 0
    for filename in fnmatch.filter(
        next(os.walk(comb_dir))[2], "route*_pts.csv"
    ):
        filepath = os.path.join(comb_dir, filename)
        if os.path.isfile(filepath):

            print(
                f"""solve_pickups_and_dropoffs :: running on
                {filename} for {cfg["sim_duration"]}s"""
            )
            name = filename[:-8]
            print("solve_pickups_and_dropoffs :: ")

            df_path = filepath
            dists_path = os.path.join(comb_dir, name + "_dists.csv")

            num_preceding_routes += solve_and_save(
                path_locations_df=df_path,
                path_distance_matrix=dists_path,
                num_vehicles=int(cfg["num_vehicles"]),
                vehicle_capacity=int(cfg["vehicle_capacity"]),
                num_seconds=int(cfg["sim_duration"]),
                capacity=cfg["capacity_col"],
                depot_index=int(cfg["depot_index"]),
                output_path=comb_routes_dir,
                num_preceding_routes=num_preceding_routes,
            )


if __name__ == "__main__":
    solve_pickups_and_dropoffs()
