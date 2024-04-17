
# Import necessary libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import utm
import contextily

def dbscan_clustering(data, eps=300, min_samples=10, plot=False):
    """
    Perform DBSCAN clustering on provided geographic data.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the longitude and latitude columns.
    - eps (float): The maximum distance (in meters) between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - plot (bool): If True, plot the clustering result on a map.
    
    Returns:
    - pd.Series: Cluster labels for each point in the DataFrame.
    """
    # Convert DataFrame to GeoDataFrame with longitude and latitude
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude),
        crs='EPSG:4326'
    )

    # Convert from geographic to UTM coordinates
    # Determine the median zone for the dataset
    median_zone_number = int(gdf['geometry'].apply(lambda x: utm.latlon_to_zone_number(x.y, x.x)).median())
    utm_crs = f'+proj=utm +zone={median_zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    gdf = gdf.to_crs(utm_crs)

    # Perform DBSCAN clustering
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto', metric='euclidean')
    labels = clusterer.fit_predict(gdf['geometry'].apply(lambda p: [p.x, p.y]).tolist())
    labels_series = pd.Series(labels, index=data.index)
    
    # Plotting
    if plot:
        f, ax = plt.subplots(1, figsize=(16, 12), dpi=100)
        
        # Subset points that are not part of any cluster (noise)
        noise = gdf.loc[labels_series == -1, 'geometry']
        clusters = gdf.loc[labels_series != -1, 'geometry']
        
        # Plot clusters
        scatter = ax.scatter(clusters.x, clusters.y, c=labels_series[labels_series != -1], s=5, cmap='viridis', linewidth=0)
        # Plot noise in grey
        ax.scatter(noise.x, noise.y, color='grey', s=5, linewidth=0)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster Label')
        
        # Add basemap (need to transform back to EPSG:4326 for contextily)
        ax.set_axis_off()
        contextily.add_basemap(ax, source=contextily.providers.CartoDB.Positron, crs=gdf.crs.to_string())
        plt.show()

    return labels_series


if __name__ == "__main__":
    data = pd.read_csv("notebooks/utils/foot-traffic/hilo_processed.csv")
    data_cleaned = data.dropna(subset=['longitude', 'latitude'])
    dbscan_clustering(data_cleaned)