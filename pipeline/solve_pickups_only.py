from pipeline.utils.cfg_parser import read_cfg
from pipeline.utils.google_cvrp import solve_and_save

if __name__ == '__main__':
    
    # read cfg
    cfg = read_cfg("../pipeline/utils/config_inputs.ini", "solve.pickups_only")

    # solve cvrp for pickups only
    solve_and_save(
        path_locations_df=cfg["truth_df_path"],
        path_distance_matrix=cfg["truth_dist_path"],
        num_vehicles=int(cfg["truth_num_vehicles"]),
        vehicle_capacity=int(cfg["vehicle_capacity"]),
        num_seconds=int(cfg["truth_sim_duration"]),
        capacity=cfg["capacity_pickup"],
        depot_index=int(cfg["depot_index"]),
        output_path=cfg["cvrp_pickups_dir"],
    )