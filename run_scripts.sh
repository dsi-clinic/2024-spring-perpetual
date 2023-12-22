#!/bin/bash

# Define file paths
indoor_outdoor_pts_csv="Galveston_data/indoor_outdoor_pts.csv"
# distance_matrix_npy="data/generated_distance_matrices/distance_matrix_${timestamp}.npy"
# capacity_list_pkl="data/capacity_lists/capacity_list_${timestamp}.pkl"
# route_list_pkl="data/generated_route_list/route_list_${timestamp}.pkl"
# distance_list_pkl="data/generated_distance_list/distance_list_${timestamp}.pkl"
# capacity_map_html="data/generated_capacity_map/${timestamp}_capacity_map.html"

# Run GenerateDistMatrix.py to generate distance matrix
echo "Running GenerateDistMatrix.py for distance matrix..."
python scripts/GenerateDistMatrix.py "$indoor_outdoor_pts_csv"

# Run GetCapacityList.py to generate capacity list
echo "Running GetCapacityList.py for capacity list..."
python GetCapacityList.py "$indoor_outdoor_pts_csv"

# Run CapacityRouting.py
echo "Running CapacityRouting.py..."
python CapacityRouting.py "$distance_matrix_npy" "$capacity_list_pkl"

# Run BuildCapacityMap.py
echo "Running BuildCapacityMap.py..."
python BuildCapacityMap.py "$indoor_outdoor_pts_csv" "$route_list_pkl" 

echo "Scripts executed successfully."
