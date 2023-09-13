"""Simple Vehicles Routing Problem (VRP).
   Distances are in meters.
"""

# Get routing configuration
import configparser
config = configparser.ConfigParser()
config.read('config_routing.ini')

import ast
TOTAL_VEHICLES = ast.literal_eval(config['cvrp']['TOTAL_VEHICLES'])
MAX_DISTANCE = ast.literal_eval(config['cvrp']['MAX_DISTANCE'])
MULTIPLIER = ast.literal_eval(config['cvrp']['MULTIPLIER'])
VEHICLE_CAPACITIES = ast.literal_eval(config['cvrp']['VEHICLE_CAPACITIES'])

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pickle
import sys
import datetime
import numpy as np

def create_data_model(matrix_file, capacity_file):
    """Stores the data for the problem."""

    data = {}
    # The matrix is divided by 10 for ease of calculation
    data['distance_matrix'] = (np.load(matrix_file)/MULTIPLIER).astype(int)
    data['demands'] = np.load(capacity_file, allow_pickle=True)
    data['num_vehicles'] = int(TOTAL_VEHICLES)
    data["vehicle_capacities"] = VEHICLE_CAPACITIES
    data['depot'] = 0
    return data


def save_to_table(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    routes = []
    distances = []
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        route = []
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            route.append({node_index:route_load})
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        route.append({manager.IndexToNode(index):route_load})
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        #print(plan_output)
        total_distance += route_distance
        total_load += route_load
        routes.append(route)
        distances.append(route_distance)
    return routes, distances

# Function below is from Google OR-Tools VRP example

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data["demands"][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")


def main():
    """Entry point of the program."""

    filename = sys.argv[1]
    file2 = sys.argv[2]

    # Instantiate the data problem.
    data = create_data_model(filename, file2)
    data['distance_matrix'] = data['distance_matrix'].tolist()

    # print(type(data['distance_matrix']))
    # print(data['demands'])
    # print(data['num_vehicles'])
    # print(data['vehicle_capacities'])
    # print(data['depot'])
    # print(filename)
    # print(file2)

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
    search_parameters.time_limit.FromSeconds(600)

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Get timestamp for output file
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y%m%d_%H%M%S')

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        route_list, distance_list = save_to_table(data, manager, routing, solution)
        print(route_list, distance_list)
        route_output = f"data/generated_route_list/CAProute_list_{timestamp_str}.pkl"
        distance_output = f"data/generated_distance_list/CAPdistance_list_{timestamp_str}.pkl"
        with open(route_output, 'wb') as f:
            pickle.dump(route_list, f)
        with open(distance_output, 'wb') as f:
            pickle.dump(distance_list, f)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()