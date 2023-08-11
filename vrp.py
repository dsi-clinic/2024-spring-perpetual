"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

TOTAL_VEHICLES = 1
MAX_DISTANCE = 30000
MULTIPLIER = 10

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import utils.data_creator as data_creator
import pickle
import sys
import datetime

# Function added to save the routes as a list
def save_to_table(data, manager, routing, solution):
    """Saves as a list."""
    routes = []
    distances = []
    for vehicle_id in range(routing.vehicles()):
        route_distance = 0
        index = routing.Start(vehicle_id)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
            route.append(manager.IndexToNode(index))
        routes.append(route)
        distances.append(route_distance)

    # Multiply the distances by 10 to get the actual distance
    distances = distances * MULTIPLIER
    return routes, distances


# Function below is from Google OR-Tools VRP example

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f'Objective: {solution.ObjectiveValue()}')
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))


def main():
    """Entry point of the program."""

    filename = sys.argv[1]

    # Instantiate the data problem.
    data = data_creator.create_data_model(filename, number_of_vehicles=TOTAL_VEHICLES)
    data['distance_matrix'] = data['distance_matrix'].tolist()

    #data = example_data.create_data_model()

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

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        MAX_DISTANCE,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.log_search = True
    #search_parameters.time_limit.seconds = 30

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Get timestamp for output file
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime('%Y%m%d_%H%M%S')

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        route_list, distance_list = save_to_table(data, manager, routing, solution)
        print(route_list, distance_list)
        route_output = f"data/route_list_{timestamp_str}.pkl"
        distance_output = f"data/distance_list_{timestamp_str}.pkl"
        with open(route_output, 'wb') as f:
            pickle.dump(route_list, f)
        with open(distance_output, 'wb') as f:
            pickle.dump(distance_list, f)
    else:
        print('No solution found !')


if __name__ == '__main__':
    main()