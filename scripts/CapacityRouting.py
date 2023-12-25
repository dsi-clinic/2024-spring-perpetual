"""
Capacitated Vehicles Routing Problem (CVRP).
Distances are in meters.

References:
OR-Tools CP-SAT v9.8. Laurent Perron and Frédéric Didier. 
https://developers.google.com/optimization/cp/cp_solver.   

"""
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pickle
import sys
import datetime
import numpy as np
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from utils import utils

def read_config(root_dir):
    """Reads config.ini file and returns routing configuration"""

    # Get routing configuration
    import configparser
    config = configparser.ConfigParser()
    config_path = os.path.join(root_dir, 'config.ini')
    config.read(config_path)

    import ast
    total_vehicles = ast.literal_eval(config['cvrp']['TOTAL_VEHICLES'])
    multiplier = ast.literal_eval(config['cvrp']['MULTIPLIER'])
    vehicle_capacities = ast.literal_eval(config['cvrp']['VEHICLE_CAPACITIES'])
    solver_time_limit = ast.literal_eval(config['cvrp']['SOLVER_TIME_LIMIT'])

    return total_vehicles, multiplier, vehicle_capacities, solver_time_limit

def create_data_model(matrix_file, capacity_file, total_vehicles, vehicle_capacities, multiplier=1):
    """Stores the data for the problem."""

    data = {}
    # Note: The matrix is divided by the MULTIPLIER as it 
    # reduces calculation time with rounded coordinates
    data['distance_matrix'] = (np.load(matrix_file)/multiplier).astype(int)
    data['demands'] = np.load(capacity_file, allow_pickle=True)
    data['num_vehicles'] = int(total_vehicles)
    data["vehicle_capacities"] = vehicle_capacities
    data['depot'] = 0
    return data

# Functions below are derived from Google OR-Tools CVRP example

def capacitated_routing(data, solver_time_limit=60):
    
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.log_search = True
    search_parameters.time_limit.FromSeconds(solver_time_limit)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    return solution, manager, routing


def get_file_paths():
    """
    Retrieves file paths from command line arguments.

    :return: A tuple containing the matrix file path and the capacity file path.
    :raises IndexError: If the required arguments are not provided.
    """
    if len(sys.argv) < 3:
        raise IndexError("Not enough arguments provided. Please provide a matrix file and a capacity file.")

    matrix_file = sys.argv[1]
    capacity_file = sys.argv[2]

    return matrix_file, capacity_file


def main():
    """Entry point of the program."""

    # Construct the path to the required directories
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.join(current_dir, '..')
    data_dir = os.path.join(root_dir, 'data')

    # Get file paths
    matrix_file, capacity_file = get_file_paths()
    matrix_file = os.path.join(data_dir, matrix_file)
    capacity_file = os.path.join(data_dir, capacity_file)

    # Get routing configuration
    try:
        total_vehicles, multiplier, vehicle_capacities, solver_time_limit = read_config(root_dir)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found.")

    # Instantiate the data problem.
    data = create_data_model(matrix_file, capacity_file, total_vehicles, vehicle_capacities, multiplier)
    data['distance_matrix'] = data['distance_matrix'].tolist()

    # Solve the problem.
    solution, manager, routing = capacitated_routing(data, solver_time_limit)

    # Get timestamp for output file
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y-%m-%d_%H:%M')

    # Print solution on console.
    if solution:
        utils.print_solution(data, manager, routing, solution)
        route_list, distance_list = utils.save_to_table(data, manager, routing, solution)
        route_output = f"{data_dir}/generated_route_list/route_list_{timestamp_str}.pkl"
        distance_output = f"{data_dir}/generated_distance_list/distance_list_{timestamp_str}.pkl"
        with open(route_output, 'wb') as f:
            pickle.dump(route_list, f)
        with open(distance_output, 'wb') as f:
            pickle.dump(distance_list, f)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()