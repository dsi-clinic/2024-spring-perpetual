#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:04:51 2024

@author: genie_god
"""


from gurobipy import Model, GRB, quicksum
import pandas as pd


def gurobi_cvrp(Q, A, N, c, V, q):
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
    mdl.Params.TimeLimit = 100  # seconds
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
    c = {(i, j): distance.iloc[i, j] for i, j in A}
    # Q: number of vehicle
    Q = 20
    # q pickup/dropof capacity for each vertix
    capacity_pickup = data['Daily_Pickup_Totes']
    q = {i: capacity_pickup.iloc[i] for i in N}
    return N, V, A, Q, q, c


def print_route(active_arcs):
    pairs_starting_with_0 = [(i, j) for i, j in active_arcs if i == 0]

    for start in pairs_starting_with_0:
        active_arcs.remove(start)

    route_num = 1
    for start in pairs_starting_with_0:
        route = [0]
        route = trace_route(route, start[1], list(active_arcs))
        print(f"Route{route_num}:", " -> ".join(map(str, route)))
        route_num += 1


def main():
    gal_points = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_pts_galv.csv")
    gal_dis = pd.read_csv(
        "../archive/2023-fall-clinic/data/indoor_outdoor_distances_galv.csv")

    N, V, A, Q, q, c = get_model_figure(gal_points, gal_dis)

    active_arcs = gurobi_cvrp(Q, A, N, c, V, q)

    print_route(active_arcs)


if __name__ == "__main__":
    main()
