This is the data dictionary for the Feasibility File, which records the inputs and outputs from each routing model

`trial_name`: str - name of the model; might correspond to the folder where the routes are saved in output/routes

`description`: str - brief text description of what the model is for

`path_to_distance_matrix`: str - the file path to the distance matrix used in the model; any path that contains "archive" in its file path can be assumed to be an old model

`path_to_dataframe`: str - the file path to the FUE/locations dataframe used in the model; any path that contains "archive" in its file path can be assumed to be an old model

`output_path`: str - the file path where the route csv files are going to be saved for this model

`simulation_run_time`: int - how long the simulation ran in order to produce optimized results

`vehicle_type`: str - Truck or Bike vehicle used in this routing

`num_vehicles`: int - number of vehicle included in the model (can also be interpretted as number of routes; i.e. 2 vehicles could mean 2 vehicles running 1 route each, or 1 vehicle running 2 routes)

`vehicle_capacity`: int - the vehicle capacity that we inputted into the routing algorithm (will change depending on pickup or dropoff, bike or truck)

`total_num_locations`: int - the total number of service locations in the input locations dataframe

`cumulative_load`: int - the total number of totes to be serviced in the input locations dataframe (i.e. total number of pickup totes or total number of dropoff totes)

`cumulative_time`: int - the total time it will take all routes to complete

`cumulative_cost`: int - the total cost of servicing all the routes

`time_per_vehicle`: list of int - list of the time it will take for each vehicle/route service (for example [time for route 1, time for route 2])

`cost_per_vehicle`: list of int - list of the cost it will take for each vehicle/route service (for example [cost of route 1, cost of route 2])

`load_per_vehicle`: list of int - list of the total tote load that each vehicle/road will complete in its service (for example [load for vehicle 1, load for vehicle 2]; might be dirty or clean bins depending on if it is a dropoff or pickup route)

`locations_per_vehicle`: list of int - list of the number of locations each vehicle/route will service (for example [number of locations serviced in route 1, number of locations serviced in route 2])

`distance_per_vehicle`: list of int - list of the total distance each vehicle/route will travel (for example [total distance of route 1, total distance of route 2])

`path_to_visualizations`: str - the file path to the visualizations of the routing 

`name_of_visualizations`: str - the name of the visualization file

`cumulative_distance`: int - the total distance of servicing all the routes

`capacity_type`: str - Daily_Pickup_Ttoes or Weekly_Dropoff_Totes (specifies if the model is for pickup or dropoff service)