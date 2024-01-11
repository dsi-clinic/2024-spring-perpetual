#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:24:39 2023

@author: genie_god
"""
# from pathlib import Path
from GenerateDistMatrix import generate_distance_matrix
from GenerateSingleSourceofTruth import generate_single_source_of_truth
from GetWayPoint import add_way_points_to_route
from RouteWithTime import add_time_to_route

if __name__ == "__main__":
    # prepare data for CVRP/Gurobi
    generate_distance_matrix()
    generate_single_source_of_truth()

    # Calculating best route

    # prepare route data for visualization
    add_way_points_to_route()
    add_time_to_route()

    # visualization
