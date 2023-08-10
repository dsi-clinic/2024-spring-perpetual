# 2023-clinic-perpetual
Fall clinic project for Perpetual
# List of Files and Their Functions

**hilo_google_api_distirbution.ipynb**
- Downloads data on businesses within Hilo, HI that might have use for reusable foodware using the Google API
- Produces “hilo_common_dist.csv” and “hilo_uncommon_dist.csv”

**hilo_google_api_collection.ipynb**
- Downloads data on businesses within Hilo, HI that might be suitable places for collection points using the Google API
- Produces “hilo_tourist_collection.csv” and “hilo_residential_collection.csv”

**Galv_parcel_mapping.ipynb**
- Draws parcel map for galveston, with colors showing different types of land use
- Recodes land use type into our categories of interest
- Produces 'Galv_Parcel_Complete.geojson'

**Galv_google_api_all.ipynb**
- Downloads data on businesses within Galveston, TX that might be suitable places for distribution points using the Google API
- Downloads data on businesses within Galveston, TX that might be suitable places for collection points using the Google API
- Merge business hours data to to combined distribution + collection dataframe
- Produces 'dirty_galv_uncommon_v3.csv', 'dirty_galv_common_v3.csv', 'galv_clct_uncommon_v1.csv', 'galv_clct_common_v1.csv', 'galv_dist_detail.csv'

**Merging Google and Yelp Datasets.ipynb**
- Inputs are multiple csv files that have restaurant names and locations/coordinates
- Merges the datasets together and removes strict duplicates and fuzzy duplicates
- Produces the merged dataset without duplicates

**Yelp_API_Business_Details_codes.ipynb**
- Use Yelp Business Search API to get all ID of business in Hilo and Gavleston, and then use these ID to get these two cities’ all business details
- No inputs, but it required a config file for Yelp API key that you can get it from the Yelp Developer Portal
- Produces “Hilo_Yelp.csv” and “Galveston_Yelp.csv”

**hilo_distribution_sites_weighting.ipynb**
- Uses variables obtained through the Yelp and Google APIs as proxy variables for foodware usage volume for FUEs in the distribution data for Hilo
- Variables are used to compute a convenience score that will be later used for FUE clustering to find distribution points
-  Produces “hilo_dist_detail.csv”

**Galveston_distribution_kmean_60.ipynb (Or 125.ipynb)**
- Use the weighted distribution sites of Galveston to perform K-mean clustering and use cluster centers as the potential distribution points. Based on the potential distribution points to draw coverage areas
- Inputs: “galv_dist_detail.csv”
- Produces “galveston_distribution_centroid_60.geojson” (potential distribution point) and “galveston_distribution_buffer_60.geojson” (coverage area of distribution point)
- For 125 collections points, changing all numbers from 60 to 125

**Hilo_distribution_kmean_80.ipynb (Or 125.ipynb)**
- Use the weighted distribution sites of Hilo to perform K-mean clustering and use cluster centers as the potential distribution points. Based on the potential distribution points to draw coverage areas
- Inputs: “hilo_dist_detail.csv”
- Produces “hilo_distribution_centroid_60.geojson” (potential distribution point) and “hilo_distribution_buffer_60.geojson” (coverage area of distribution point)
- For 125 collections points, changing all numbers from 80 to 125

**Hilo_kmean_collection_points_60_points.ipynb (Or 125.ipynb)**
- Assign weights to Hilo Parcel data and perform k-mean clustering for the parcels to determine the theoretical positions of collection points and produce their coverage areas
- Inputs: “Zoning_(Hawaii_County).shp” and “All_Hilo_Business.csv”
- Produces “Hilo_parcel_cluster_center_60.geojson” (locations of theoretical collection points) and “Hilo_parcel_cluster_buffer_60.geojson” (their coverage areas). Since they are the inputs of other ipynb, you can find these two geojson in the data folder, not the output folder.
- For 125 collections points, changing all numbers from 60 to 125

**Galveston_kmean_collection_points_final_60_point.ipynb (Or 125.ipynb)**
- Assign weights to Galveston Parcel data and perform k-mean clustering for the parcels to determine the theoretical positions of collection points and produce their coverage areas
- Inputs: “Galveston_Parcel_Complete.geojson” and “All_Galveston_Business.csv”
- Produces “galveston_parcel_cluster_center_60.geojson” (locations of theoretical collection points) and “galveston_parcel_cluster_buffer_60.geojson”) (their coverage areas) . Since they are the inputs of other ipynb, you can find these two geojson in the data folder, not the output folder.
- For 125 collections points, changing all numbers from 60 to 125

**skater_for_hilo_70.ipynb (Or 140.ipynb)**
- Perform alternative clustering methods (skater algorithm) on weighted Hilo parcel data created before to have clustered parcels and cluster points that help us verify the results of K-mean clustering. Create coverage areas for collection points
- Skater algorithm does not allow us to control the exact number of clusters we want. Based on my experience, the number of clusters of the outputs is higher than the number of clusters I gave to the algorithm
- Check https://pysal.org/spopt/notebooks/skater.html for more explanation
- Inputs:”hilo_parcel_skater.shp”
- Produces “hilo_cluster_skater_70.geojson” (clustered parcels), “hilo_cluster_centroid_skater_70.geojson” (collection points), and “hilo_cluster_buffer_skater_70.geojson” (coverage areas)
- For 140 collections points, changing all numbers from 70 to 140

**skater_for_galveston.ipynb (Or 140.ipynb)**
- Perform alternative clustering methods (skater algorithm) on weighted Galveston parcel data created before to have clustered parcels and cluster points that help us verify the results of K-mean clustering. Create coverage areas for collection points
- Skater algorithm does not allow us to control the exact number of clusters we want. Based on my experience, the number of clusters of the outputs is higher than the number of clusters I gave to the algorithm
- For Galveston, the algorithm takes extremely long time, so it is recommended to use high-speed computing server for this file.
- Check https://pysal.org/spopt/notebooks/skater.html for more explanation
- Inputs:”galveston_parcel_skater.shp”
- Produces “gal_cluster_skater_70.geojson” (clustered parcels), “gal_cluster_centroid_skater_70.geojson” (collection points), and “gal_cluster_buffer_skater_70.geojson” (coverage areas). Theoretically, the codes should work, but I haven’t finished running it yet. Therefore, the outputs are not in the “output” folder at this moment.
- For 140 collections points, changing all numbers from 70 to 140

**hilo_collection_sites.ipynb**
- Finds the closest business to the center of each parcel cluster in Hilo
- Produces “hilo_cluster_collection_sites_kmeans_60.csv”, “hilo_cluster_collection_sites_60.geojson”, “hilo_cluster_collection_sites_buffer_60.geojson”, “hilo_cluster_collection_sites_kmeans_120.csv”, “hilo_cluster_collection_sites_120.geojson”, “hilo_cluster_collection_sites_buffer_120.geojson”, “hilo_cluster_collection_sites_skater_70.csv”, “hilo_cluster_collection_sites_skater_70.geojson”, “hilo_cluster_collection_sites_buffer_skater_70.geojson”, “hilo_cluster_collection_sites_skater_140.csv”, “hilo_cluster_collection_sites_skater_140.geojson”, and “hilo_cluster_collection_sites_buffer_skater_140.geojson”

**Calculating_Distance_Clct_Galveston_60.ipynb (Or 125.ipynb)**
- Finds the closest business to the center of each parcel cluster in Galveston (the real collection points)
- Produces “galv_cluster_collection_sites_60.csv”, “galv_cluster_collection_sites_buffer_60.geojson”.
- For 125 collections points, changing all numbers from 60 to 125

**Truck Routes.ipynb**
-inputs are a csv file with location coordinates, the number of trucks you’d like to use, and a central point
- produces truck routes that minimize distance for each truck

**foot_traffic_mapping.ipynb**
- Draws parcel maps for Galveston, TX and Hilo, HI

# Interactive Map Links:

**Distribution Map:**
- Galveston (60 distribution points): https://arcg.is/1q1X5L0
- Galveston (125 distribution points): https://arcg.is/1r1Wby 
- Hilo (80 distribution points): https://arcg.is/1fi8rX0 
- Hilo (125 distribution points): https://arcg.is/bb954 

**Collection Map:**
- Galveston (60 collection points): https://arcg.is/0044L8 
- Galveston (125 collection points): https://arcg.is/iHeXD 
- Hilo (60 collection points): https://arcg.is/1Lz8XT1
- Hilo (125 collection points): https://arcg.is/0raSmP  
- Hilo (71 collection points using Skater)  https://arcg.is/0j4mfu   
- Hilo (126 collection points using Skater) https://arcg.is/04yiDn2


