"""Route optimization algorithms provided by Gurobi under an academic license.
"""

# Standard library imports
from typing import List, Tuple

# Third-party imports
import pandas as pd
from gurobipy import Model, GRB, quicksum

# Application imports
from pipeline.routes.common import IRoutingClient, SimulationParams


class GurobiClient(IRoutingClient):
    """A wrapper class for the Gurobi Optimizer."""

    def _parse_solution(
        self, active_arcs: List[Tuple[int, int]], locations_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Builds a DataFrame of routes from a list of arcs.

        Args:
            active_arcs (`list` of (`int`, `int`)): A list of two-item
                tuples holding the indices of location pairs that are
                traveled in the optimal route solution (e.g., [(50, 10), ...]).

            locations_df (`pd.DataFrame`): The locations used for routing.

        Returns:
            (`pd.DataFrame`): A DataFrame containing the optimal routes.
        """
        # Find the first location for each route following the depot
        pairs_starting_with_0 = [(i, j) for i, j in active_arcs if i == 0]

        # Remove depot from each arc to avoid incorrect arc traversal
        for start in pairs_starting_with_0:
            active_arcs.remove(start)

        # Build DataFrame of routes
        route_counter = 0
        selected_arcs = list(active_arcs)
        routes_df = None
        for start in pairs_starting_with_0:

            # Build single route from arc tuples
            route = [0, start[1]]
            while True:
                found_next = False
                for arc in selected_arcs:
                    if arc[0] == route[-1]:
                        route.append(arc[1])
                        selected_arcs.remove(arc)
                        found_next = True
                        break
                if not found_next:
                    break

            # Convert route to DataFrame
            df = pd.DataFrame(route, columns=["Node"])
            df["Route"] = route_counter

            # Append to existing DataFrame containing all routes
            routes_df = df if routes_df is None else pd.concat([routes_df, df])

            # Iterate counter
            route_counter += 1

        # Merge DataFrame with original locations
        merged_df = (
            routes_df.merge(
                right=locations_df.reset_index(),
                how="left",
                left_on="Node",
                right_on="index",
            )
            .rename(columns={"Node": "Original Index"})
            .drop(columns="index")
        )

        return merged_df

    def solve_bidirectional_cvrp(
        self,
        locations_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        pickup_params: SimulationParams,
        combo_params: SimulationParams,
    ) -> pd.DataFrame:
        """Solves a subcase of the Capacitated Vehicle Routing Problem (CVRP)
        in which trucks can pick up and drop off items at the same locations.

        Args:
            locations_df (`pd.DataFrame`): The locations to use for routing.

            distance_df (`pd.DataFrame`): A matrix of distances computed between
                every unique location in `locations_df`. Units are expressed in meters.

            pickup_params (`SimulationParams`): Configuration for a pickup route
                simulation.

            combo_params (`SimulationParams`): Configuration for a combined
                pickup/drop off route simulation.

        Returns:
            (`pd.DataFrame`): A DataFrame containing the optimal routes.
        """
        # Reassign vehicle capacity to new variable
        Q = combo_params.vehicle_capacity

        # Compute indices of all locations
        N = [i for i in range(1, len(locations_df))]
        V = [0] + N

        # Create list of arcs (pairs of unique locations)
        A = [(i, j) for i in V for j in V if i != j]

        # Store distance between each arc
        c = {(i, j): distances_df.iloc[i, j] for i, j in A}

        # Create dictionary of pickup/drop-off location capacities
        capacity_pickup = locations_df[combo_params.capacity_col]
        q = {i: capacity_pickup.iloc[i] for i in N}

        # Initialize CVRP model
        mdl = Model("CVRP")

        # Add variable for location index
        x = mdl.addVars(A, vtype=GRB.BINARY)

        # Add variable of number of client to the model
        u = mdl.addVars(N, vtype=GRB.CONTINUOUS)

        # Set the model goal (minimizing the total distance)
        mdl.modelSense = GRB.MINIMIZE
        mdl.setObjective(quicksum(x[i, j] * c[i, j] for i, j in A))
        mdl.addConstrs(quicksum(x[i, j] for j in V if j != i) == 1 for i in N)
        mdl.addConstrs(quicksum(x[i, j] for i in V if i != j) == 1 for j in N)
        mdl.addConstrs(
            (x[i, j] == 1) >> (u[i] + q[j] == u[j]) for i, j in A if i != 0 and j != 0
        )

        # Add constraints to ensure that each location's load is not negative
        mdl.addConstrs(u[i] + q[i] >= 0 for i in N)
        mdl.addConstrs(u[i] <= Q for i in N)

        # Configure minimum gap value Gurobi has to reach before declaring optimality
        mdl.Params.MIPGap = 0.1

        # Configure number of seconds to run model simulation
        mdl.Params.TimeLimit = combo_params.runtime

        # Find solution
        mdl.optimize()

        # Process arcs to find solution
        active_arcs = [a for a in A if x[a].x > 0.999]

        # Convert arc routes to DataFrame
        route_locations_df = self._parse_solution(active_arcs, locations_df)

        return route_locations_df
