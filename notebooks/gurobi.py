#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:04:51 2024

@author: genie_god
"""


from gurobipy import Model, GRB, quicksum
import pandas as pd
import os


def gurobi_cvrp(Q, A, N, c, V, q, running_time):
    mdl = Model('CVRP')
    # adding variable of location index
    x = mdl.addVars(A, vtype=GRB.BINARY)
    # adding variable of number of client to the model
    u = mdl.addVars(N, vtype=GRB.CONTINUOUS)
    # setup the goal of this model going to achieve
    # minimize the total distance
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[i, j]*c[i, j] for i, j in A))
    mdl.addConstrs(quicksum(x[i, j] for j in V if j != i) == 1 for i in N)
    mdl.addConstrs(quicksum(x[i, j] for i in V if i != j) == 1 for j in N)
    mdl.addConstrs((x[i, j] == 1) >> (u[i]+q[j] == u[j])
                   for i, j in A if i != 0 and j != 0)
    # mdl.addConstrs(u[i] >= q[i] for i in N)
    # might work this way for both pickup and drop off,
    # add constraints that each points's load can not be negative
    mdl.addConstrs(u[i] + q[i] >= 0 for i in N)
    mdl.addConstrs(u[i] <= Q for i in N)
    mdl.Params.MIPGap = 0.1
    mdl.Params.TimeLimit = running_time  # seconds
    mdl.optimize()

    active_arcs = [a for a in A if x[a].x > 0.999]
    return active_arcs


def trace_route(route, start, selected_arcs):
    route.append(start)
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
    return route


def get_model_figure(data, distance):
    # number of location includes depot
    n = data.shape[0]

    # N is the list of points (pickup and drop off)
    N = [i for i in range(1, n)]
    # V all the vertices including depot
    V = [0] + N

    # arc
    A = [(i, j) for i in V for j in V if i != j]
    # arc with distance
    c = {(i, j): distance.iloc[i, j] for i, j in A}
    # q pickup/dropof capacity for each vertix
    capacity_pickup = data['Daily_Pickup_Totes']
    q = {i: capacity_pickup.iloc[i] for i in N}
    return N, V, A, q, c


def print_save_route(active_arcs):
    pairs_starting_with_0 = [(i, j) for i, j in active_arcs if i == 0]

    for start in pairs_starting_with_0:
        active_arcs.remove(start)

    route_num = 1
    routes_dataframes = {}
    clear_routes_csv()
    for start in pairs_starting_with_0:
        route = [0]
        route = trace_route(route, start[1], list(active_arcs))
        print(f"Route{route_num}:", " -> ".join(map(str, route)))

        # Convert route to DataFrame and save in the dictionary
        df = pd.DataFrame(route, columns=['Node'])
        routes_dataframes[f'Route{route_num}'] = df

        # Save DataFrame to CSV
        csv_filename = f'routes/route_{route_num}.csv'
        df.to_csv(csv_filename, index=True)
        print(f"Route{route_num} saved to {csv_filename}")

        route_num += 1

    return routes_dataframes


def clear_routes_csv(directory_path='routes'):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # List all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)
            # Delete the file
            os.remove(file_path)
            print(f"Deleted {file_path}")


def main():
    gal_points = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_pts_galv.csv")
    gal_dis = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_distances_galv.csv")

    N, V, A, q, c = get_model_figure(gal_points, gal_dis)
    vehicle_capacity = 100
    runnig_time = 3600 * 2
    active_arcs = gurobi_cvrp(vehicle_capacity, A, N, c, V, q, runnig_time)

    print_save_route(active_arcs)


if __name__ == "__main__":
    main()
