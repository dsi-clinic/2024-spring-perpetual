### Data

`galveston_indoor_pts.csv`: This file contains all the FUEs that will serve as both pickup and dropoff locations for the simulation. Perpetual refers to them as indoor points.

`galveston_outdoor_pts.csv`: This file contains all the locations where Perpetual will place a bin. Perpetual refers to them as outdoor points and will only be used for pickup. This dataset also marks whether a location is going to be served by a bike or a truck. The following is the classification for the id of the bins:
- city approved have IDs starting from 1000
- private parking bins have IDs starting from 2000
- park_board bins have IDs starting from 3000

`indoor_outdoor_pts_galv.csv`: csv of concatonated indoor and outdoor service locations in Galveston

`indoor_outdoor_distances_galv.csv`: csv of the distance matrix for all indoor and outdoor points in galveston (converted from the npy file "distance_matrix_new.npy")

`generated_distance_matrices` : folder that contains the distance matrices generated for galveston data 
- `distance_matrix_20231029_220129.npy`: updated distance matrix with both indoor and outdoor points
- `distance_matrix_new.npy`: updated distance matrix with both indoor and outdoor points (renamed for ease)
