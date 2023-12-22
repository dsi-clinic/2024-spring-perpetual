#!/bin/bash

# Load config file
config_file="config.ini"

# Check if the config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file not found!"
    exit 1
fi

# Read file paths from the config file
location_data=$(grep '^location_data=' "$config_file" | cut -d'=' -f2)
distance_matrix=$(grep '^distance_matrix=' "$config_file" | cut -d'=' -f2)
capacity_list=$(grep '^capacity_list=' "$config_file" | cut -d'=' -f2)
route_list=$(grep '^route_list=' "$config_file" | cut -d'=' -f2)

# Verify the paths (Optional)
if [ ! -f "$location_data" ]; then
    echo "Warning: File at location_data not found"
fi

if [ ! -f "$distance_matrix" ]; then
    echo "Warning: File at distance_matrix not found"
fi

if [ ! -f "$capacity_list" ]; then
    echo "Warning: File at capacity_list not found"
fi

if [ ! -f "$route_list" ]; then
    echo "Warning: File at route_list not found"
fi

# Print the list of tasks
echo "Before selecting the stage ahead, please make sure that the correct file paths have been specified in config.ini."
echo "Select a stage to run:"
echo "1. Stage 1: Generate input parameters for routing"
echo "2. Stage 2: Run routing algorithm"
echo "3. Stage 3: Build capacity map"

# Get user input
read -p "Enter the number of the task you want to run: " task

# Execute the chosen task
case $task in
    1)
        echo "Running Stage 1..."
        filepath="Galveston_data/location_data.csv"

        # Run GetCapacityList.py to generate capacity list
        echo "Generating Capacity List..."
        python scripts/GetCapacityList.py "$location_data"

        # Run GenerateDistMatrix.py to generate distance matrix
        echo "Generating Distance Matrix..."
        python scripts/GenerateDistMatrix.py "$location_data"
        
        echo "The distance matrix and capacity list have been generated."
        ;;
    2)
        echo "Running Stage 2..."
        # Run CapacityRouting.py
        echo "Running CapacityRouting.py..."
        python scripts/CapacityRouting.py "$distance_matrix" "$capacity_list"
        echo "The route and distance lists have been generated and saved in the /data directory."


        ;;
    3)
        echo "Running Stage 3..."
        # Run BuildCapacityMap.py
        echo "Running BuildCapacityMap.py..."
        python scripts/BuildCapacityMap.py "$location_data" "$route_list" 
        echo "The capacity map has been generated and saved in the /outputs directory."
        ;;
    *)
        echo "Invalid selection. Please choose 1, 2, or 3."
        ;;
esac