from pipeline.utils.cfg_parser import read_cfg
from pipeline.utils.google_cvrp import solve_and_save

if __name__ == "__main__":

    # read cfg
    cfg = read_cfg("../pipeline/utils/config_inputs.ini", "solve.pickups_only")

    print(
        f"""solve_pickups_only :: running on
        {cfg["truth_df_path"]} for {cfg["sim_duration"]}s"""
    )

    # solve cvrp for pickups only
    solve_and_save(
        path_locations_df=cfg["truth_df_path"],
        path_distance_matrix=cfg["truth_dists_path"],
        num_vehicles=int(cfg["num_vehicles"]),
        vehicle_capacity=int(cfg["vehicle_capacity"]),
        num_seconds=int(cfg["sim_duration"]),
        capacity=cfg["capacity_col"],
        depot_index=int(cfg["depot_index"]),
        output_path=cfg["cvrp_pickups_dir"],
        num_preceding_routes=0
    )
