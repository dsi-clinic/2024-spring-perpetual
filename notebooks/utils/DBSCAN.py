# Import necessary libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import utm
import contextily
import os

def perform_dbscan(data, eps, min_samples):
    """
    Perform DBSCAN clustering on provided geographic data.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the longitude and latitude columns.
    - eps (float): The maximum distance (in meters) between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    
    Returns:
    - gpd.GeoDataFrame: GeoDataFrame with cluster labels and geometry.
    """
    # Convert DataFrame to GeoDataFrame with longitude and latitude
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude),
        crs='EPSG:4326'
    )

    # Convert from geographic to UTM coordinates
    median_zone_number = int(gdf['geometry'].apply(lambda x: utm.latlon_to_zone_number(x.y, x.x)).median())
    utm_crs = f'+proj=utm +zone={median_zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    gdf = gdf.to_crs(utm_crs)

    # Perform DBSCAN clustering
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(gdf['geometry'].apply(lambda p: [p.x, p.y]).tolist())
    
    gdf['cluster'] = labels
    
    return gdf

def plot_clusters(gdf, eps, min_samples, filename):
    """
    Plot the clustering result on a map using the GeoDataFrame.
    
    Parameters:
    - gdf (gpd.GeoDataFrame): GeoDataFrame containing 'geometry' and 'cluster' columns.
    - eps (float): The maximum distance (in meters) between two samples for one to be considered as in the neighborhood of the other.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    - filename (str): The name of the file to save the plot.
    """
    # Define the directory where the files will be saved
    save_directory = "data/foot-traffic/plots"

    # Full path for saving the file
    save_path = os.path.join(save_directory, filename)

    # Plotting
    f, ax = plt.subplots(1, figsize=(16, 12), dpi=100)
    
    # Subset points that are not part of any cluster (noise)
    noise = gdf[gdf['cluster'] == -1]
    clusters = gdf[gdf['cluster'] != -1]
    
    # Plot clusters
    scatter = ax.scatter(clusters.geometry.x, clusters.geometry.y, c=clusters['cluster'], s=5, cmap='viridis', linewidth=0)
    # Plot noise in grey
    ax.scatter(noise.geometry.x, noise.geometry.y, color='grey', s=5, linewidth=0)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Cluster Label')
    
    # Add basemap (transform back to EPSG:4326 for contextily)
    ax.set_axis_off()
    contextily.add_basemap(ax, source=contextily.providers.CartoDB.Positron, crs=clusters.crs.to_string())

    # Set the title dynamically based on eps and min_samples
    ax.set_title(f'Hilo DBSCAN Clustering (eps={eps}, min_samples={min_samples})', fontsize=15)
    
    plt.savefig(save_path)
    plt.show()

def batch_cluster_and_plot(data, eps_range, num_batches):
    """
    Execute clustering and plotting for various eps and min_samples values.
    """
    batch_size = max(int(len(data) * 0.01), 1)
    min_samples_range = range(batch_size, batch_size * (num_batches + 1), batch_size)

    for eps in eps_range:
        for min_samples in min_samples_range:
            gdf = perform_dbscan(data, eps, min_samples)
            plot_path = f'hilo_dbscan_eps{eps}_min{min_samples}.png'
            plot_clusters(gdf, eps, min_samples, plot_path)

if __name__ == "__main__":
    data = pd.read_csv("data/foot-traffic/output/hilo_seasonal_fall.csv")
    data_cleaned = data.dropna(subset=['longitude', 'latitude'])
    eps = 500
    min_samples = 20
    gdf = perform_dbscan(data_cleaned, eps, min_samples)
    filename = f'hilo_dbscan_eps{eps}_min{min_samples}.png'
    plot_clusters(gdf, eps, min_samples, filename)

    # Not functioning yet - plots a variety of graphs
    #eps_range= [500, 1000, 2000, 3000]
    #num_batches = 5
    #batch_cluster_and_plot(data_cleaned, eps_range, num_batches)