# Output Directory

* `map_bikes.html` : (code/bike_conversions.py output) folium map with bike-accessible points
* `map_trucks_and_bikes.html` : (code/bike_conversions.py output) folium map with all points
* `map_trucks.html` : (code/bike_conversions.py output) folium map with truck-accessible points

* `data/` : contains intermediary/resulting data output by code (i.e. modified or converted csv data from the single source of truth, that will be used in modeling)
    * `converted_bike_dists_galv.csv` : distance matrix for converted bike points
    * `converted_bike_service_pts.csv` : dataframe for converted bike points
    * `converted_truck_dists_galv.csv` : distance matrix for converted truck points
    * `converted_truck_service_pts.csv` : dataframe for converted truck points
    * `extracted_supp_info_dict.csv` : dataframe storing supplementary info dictionary for all points  

* `routes` : Contains folders for each model, with their routes and visualizations inside. The name of each folder corresponds with the model's trial_name on the feasibility file. 


* `feasibilityfile.csv` : Feasibility Analysis output csv file. 
* `data_dictionary.md` : Data dictionary for the feasibility file. 

