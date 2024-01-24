#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:04:51 2024

@author: genie_god
"""


from gurobipy import Model, GRB, quicksum


def build_gurobi_cvrp(number_vehicle,
                      distance_index,
                      pickup_points,
                      distance_matrix,
                      all_points,
                      capacity_list):
    mdl = Model('CVRP')
    # adding variable of location index
    x = mdl.addVars(distance_index, vtype=GRB.BINARY)
    # adding variable of number of client to the model
    u = mdl.addVars(pickup_points, vtype=GRB.CONTINUOUS)
    # setup the goal of this model going to achieve
    # minimize the total distance
    mdl.modelSense = GRB.MINIMIZE
    mdl.setObjective(quicksum(x[i, j]*distance_matrix[i, j]
                              for i, j in distance_index))
    mdl.addConstrs(quicksum(distance_matrix[i, j]
                            for j in all_points if j != i) == 1
                   for i in pickup_points)
    mdl.addConstrs(quicksum(distance_matrix[i, j]
                            for i in all_points if i != j) == 1
                   for j in pickup_points)
    mdl.addConstrs((x[i, j] == 1) >> (u[i]+capacity_list[j] == u[j])
                   for i, j in distance_index if i != 0 and j != 0)
    mdl.addConstrs(u[i] >= capacity_list[i] for i in pickup_points)
    mdl.addConstrs(u[i] <= number_vehicle for i in pickup_points)
    mdl.Params.MIPGap = 0.1
    mdl.Params.TimeLimit = 30  # seconds
    mdl.optimize()
