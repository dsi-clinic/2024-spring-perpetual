"""Route optimization algorithms provided by Google OR-Tools.
"""

# Third-party imports
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Application imports
from pipeline.routes.common import IRoutingClient, SimulationParams


class GoogleORToolsClient(IRoutingClient):
    """ """

    def parse_solution(self, data, manager, routing: pywrapcp.RoutingModel, solution):
        """Save each route to its own dataframe and print solutions on the console."""
        print(f"Objective: {solution.ObjectiveValue()}")
        routes = []
        distances = []
        loads = []
        total_distance = 0
        total_load = 0
        for vehicle_id in range(data["num_vehicles"]):
            route = []
            agg_distances = []
            truck_load = []
            index = routing.Start(vehicle_id)
            route_distance = 0
            route_load = 0
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_load += data["demands"][node_index]
                route.append(node_index)
                truck_load.append(route_load)
                agg_distances.append(route_distance)

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            # Update metrics for last stop in route
            route.append(manager.IndexToNode(index))
            truck_load.append(route_load)
            agg_distances.append(route_distance)

            # Update cumulative variables
            routes.append(route)
            distances.append(agg_distances)
            loads.append(truck_load)
            total_distance += route_distance
            total_load += route_load

        return routes, distances, loads

    def solve_cvrp(
        self,
        locations_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        num_vehicles: int,
        vehicle_capacity: int,
        num_seconds: int,
        capacity_col: str,
    ):
        """
        Solve the CVRP problem.

        Documentation:
        - [](https://developers.google.com/optimization/routing/cvrp)

        Args:

        Returns:
        """
        # Create data model
        data = {}
        data["distance_matrix"] = distances_df.to_numpy().astype(int)
        data["demands"] = locations_df[capacity_col].astype(int).tolist()
        data["num_vehicles"] = num_vehicles
        data["vehicle_capacities"] = [vehicle_capacity for _ in range(num_vehicles)]
        data["depot"] = 0

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Create and register a transit callback
        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data["demands"][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            evaluator_index=demand_callback_index,
            slack_max=0,  # null capacity slack
            vehicle_capacities=data["vehicle_capacities"],  # vehicle maximum capacities
            fix_start_cumul_to_zero=True,  # start cumul to zero
            name="Capacity",
        )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(num_seconds)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # Return solution.
        if not solution:
            return

        # Otherwise, parse solution for data
        routes, distances, loads = self.parse_solution(data, manager, routing, solution)

        # Concatenate into single DataFrame
        all_df = None
        for i in range(len(routes)):
            route_df = locations_df.loc[routes[i], :]
            route_df["Route"] = i
            route_df["Route Stop Number"] = list(range(1, len(routes[i]) + 1))
            route_df["Cumulative_Distance"] = distances[i]
            route_df["Truck_Load"] = loads[i]
            route_df = route_df.reset_index()
            route_df = route_df.rename(columns={"index": "Original_Index"})
            route_df = route_df.drop(columns=["level_0"], errors="ignore")
            all_df = route_df if all_df is None else pd.concat([all_df, route_df])

        return all_df

    def solve_bidirectional_cvrp(
        self,
        locations_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        pickup_params: SimulationParams,
        pickup_dropoff_params: SimulationParams,
    ):
        """
        NOTE: Pickup capacities are greater than or equal to zero,
        and the first location in the route (the depot) has a capacity
        equal to the total number of totes to drop off every week.
        """
        # Solve for pickup routes only
        pickups_only_df = self.solve_cvrp(
            locations_df,
            distances_df,
            pickup_params.num_vehicles,
            pickup_params.vehicle_capacity,
            pickup_params.runtime,
            pickup_params.place_capacity_column,
        )

        # Return if no solution
        if pickups_only_df is None:
            return

        # Group results by route
        pickup_routes = pickups_only_df.groupby("Route")

        # Process each route
        all_pickups_dropoffs_df = None
        for grp in pickup_routes.groups:

            # Get route as DataFrame
            pickup_df = pickup_routes.get_group(grp)

            # Extract location indices
            pickup_idx = pickup_df["Original_Index"][:-1]

            # Get subset of route locations
            pickup_locs_df = locations_df.loc[pickup_idx]

            # Add capacity column to locations
            pickup_locs_df["Capacity"] = [
                pickup_locs_df["Weekly_Dropoff_Totes"].sum()
            ] + pickup_locs_df["Daily_Pickup_Totes"].tolist()[1:]

            # Add locations with dropoffs as new dropoff sites and set capacities
            dropoff_locs_df = pickup_locs_df.query("`Weekly_Dropoff_Totes` > 0")
            dropoff_locs_df["Capacity"] = dropoff_locs_df["Weekly_Dropoff_Totes"] * -1

            # Combine pickup and dropoff locations into final DataFrame
            combined_locs_df = pd.concat([pickup_locs_df, dropoff_locs_df])

            # Get subset of distance matrix corresponding to pickup route locations
            pickup_dist_df = distances_df.loc[pickup_idx]

            # Get subset of distance matrix corresponding to dropoff route locations
            dropoff_idx = pickup_df.query("`Weekly_Dropoff_Totes` > 0")[
                "Original_Index"
            ]
            dropoff_dist_df = pickup_dist_df.loc[dropoff_idx]

            # Combine pickup and dropoff distances while maintaining square dimensions
            num_dropoff_sites = len(dropoff_idx)
            combined_dist_df = pd.concat([pickup_dist_df, dropoff_dist_df])[
                [str(i) for i in pickup_idx]
            ]
            combined_dist_df = pd.concat(
                [combined_dist_df, combined_dist_df.iloc[:, -num_dropoff_sites:]],
                axis=1,
            )

            # Reset distance matrix column names and index
            combined_dist_df.columns = [i for i in range(len(combined_dist_df.columns))]
            combined_dist_df = combined_dist_df.reset_index().drop(columns="index")

            # Reset location index
            combined_locs_df = combined_locs_df.reset_index()

            # Solve combined dropoff and pickup problem within pickup route
            pickups_dropoffs_df = self.solve_cvrp(
                combined_locs_df,
                combined_dist_df,
                pickup_dropoff_params.num_vehicles,
                pickup_dropoff_params.vehicle_capacity,
                pickup_dropoff_params.runtime,
                pickup_dropoff_params.place_capacity_column,
            )

            # Continue to next route if no solution exists
            if pickups_dropoffs_df is None:
                continue

            # Modify unique id for pickup route and sub-routes
            pickups_dropoffs_df["Route"] = (
                str(grp) + "-" + pickups_dropoffs_df["Route"].astype(str)
            )

            # Concatenate results
            all_pickups_dropoffs_df = (
                pickups_dropoffs_df
                if all_pickups_dropoffs_df is None
                else pd.concat([all_pickups_dropoffs_df, pickups_dropoffs_df])
            )

        return all_pickups_dropoffs_df
