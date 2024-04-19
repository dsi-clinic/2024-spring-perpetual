# Import necessary libraries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import utm
import hdbscan
import os
import contextily as ctx
import numpy as np

def perform_hdbscan(data, min_cluster_size, min_samples, filename, save_output=True):
    """
    Perform HDBSCAN clustering on provided geographic data and save the results.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the longitude and latitude columns.
    - min_cluster_size (int): The minimum size of clusters; not less than this number of points will form a cluster.
    - min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. This helps control the degree of noise.
    - filename (str): The name of the file to save the clusters.
    - save_output (bool): Whether to save the clustering output to a file.
    
    Returns:
    - gpd.GeoDataFrame: GeoDataFrame with cluster labels.
    """

    # Clean and sample dataset
    data = data.dropna(subset=['longitude', 'latitude'])
    data = data.sample(n=100000, random_state=42)  # Sample 100,000 points randomly

    # Convert DataFrame to GeoDataFrame with longitude and latitude
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude),
        crs='EPSG:4326'
    )

    # Convert from geographic to UTM coordinates for accurate distance measurement
    median_zone_number = int(gdf['geometry'].apply(lambda x: utm.latlon_to_zone_number(x.y, x.x)).median())
    utm_crs = f'+proj=utm +zone={median_zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    gdf = gdf.to_crs(utm_crs)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(pd.DataFrame({'x': gdf.geometry.x, 'y': gdf.geometry.y}))
    gdf['cluster'] = labels
    
    if save_output:
        # Define the directory where the files will be saved
        save_directory = "data/foot-traffic/output"
        
        # Full path for saving the file
        save_path = os.path.join(save_directory, filename)

        # Save output to GeoJSON
        gdf.to_file(save_path, driver='GeoJSON')

    return gdf


def summarize_clusters(gdf):
    """
    Summarizes the clusters in a GeoDataFrame that includes a 'cluster' label column.
    
    Parameters:
    - gdf (gpd.GeoDataFrame): GeoDataFrame with a 'cluster' column where -1 indicates noise.
    
    Returns:
    - pd.DataFrame: DataFrame where each row represents a cluster with columns for the
                    number of points in the cluster, the cluster label, and the centroid geometry.
    """
    # Filter out the noise points
    clusters = gdf[gdf['cluster'] != -1]

    # Group by cluster label
    grouped = clusters.groupby('cluster')

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'num_points': grouped.size(),  # Count points per cluster
        'centroid': grouped['geometry'].apply(lambda x: x.unary_union.centroid)  # Calculate centroid
    }).reset_index()

    # Rename columns appropriately
    summary_df.columns = ['cluster_label', 'num_points', 'geometry']

    return summary_df

def plot_cluster_centroids(cluster_summary_df):
    """
    Plots the centroids of clusters on a map, annotating each centroid with the number of points in the cluster,
    and overlays this on a basemap for geographical context.
    
    Parameters:
    - cluster_summary_df (pd.DataFrame): DataFrame with columns 'cluster_label', 'num_points', and 'geometry' (centroids).
    """
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(cluster_summary_df, geometry='geometry', crs="EPSG:4326")
    
    # Convert CRS to EPSG:3857 for contextily
    gdf = gdf.to_crs(epsg=3857)

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each centroid
    gdf.plot(ax=ax, color='blue', markersize=gdf['num_points']/10, alpha=0.6)  # Adjust marker size by number of points

    # Annotate the centroid with the number of points in the cluster
    for idx, row in gdf.iterrows():
        ax.text(row.geometry.x, row.geometry.y, f' {row["num_points"]}', fontsize=8, ha='left')

    # Add a basemap
    ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    # Set plot parameters
    ax.set_title('Cluster Centroids with Number of Points')
    ax.axis('off')  # Turn off axis for better visual appearance

    plt.show()

# group by for labels, discard clusters --> new df
# new function to restructure dataframe
# each row is a cluster (columns: num_points, cluster label, geometry)
# plot cluster centroids using new df (also number of points in each cluster, visualize diff
# change opacity etc, add noise if not too many points)
# save this algorithm output to look at

def plot_clusters(summary_df, filename):
    gdf = gpd.GeoDataFrame(summary_df, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)

    # Filtering out non-finite values
    gdf = gdf[gdf['geometry'].x.notnull() & gdf['geometry'].y.notnull()]
    gdf = gdf[(gdf['geometry'].x != np.inf) & (gdf['geometry'].x != -np.inf)]
    gdf = gdf[(gdf['geometry'].y != np.inf) & (gdf['geometry'].y != -np.inf)]

    fig, ax = plt.subplots(1, figsize=(10, 8))
    gdf.plot(ax=ax, markersize=5, color='blue')  # Adjust marker size appropriately

    # Manually setting zoom level
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=10)
    
    # Optionally setting spatial limits
    ax.set_xlim([gdf.geometry.bounds.minx.min(), gdf.geometry.bounds.maxx.max()])
    ax.set_ylim([gdf.geometry.bounds.miny.min(), gdf.geometry.bounds.maxy.max()])

    plt.savefig(filename)
    plt.show()


# Example usage (assuming 'summary_df' is already computed and available):
# plot_clusters(summary_df, 'cluster_centroids.png')

def batch_cluster_and_plot(data, eps_range, num_batches):
    """
    Execute clustering and plotting for various eps and min_samples values.
    """
    batch_size = max(int(len(data) * 0.01), 1)
    min_samples_range = range(batch_size, batch_size * (num_batches + 1), batch_size)

    for eps in eps_range:
        for min_samples in min_samples_range:
            gdf = perform_hdbscan(data, eps, min_samples)
            plot_path = f'hilo_dbscan_eps{eps}_min{min_samples}.png'
            plot_clusters(gdf, eps, min_samples, plot_path)

if __name__ == "__main__":
    data = pd.read_csv("data/foot-traffic/output/hilo_seasonal_fall.csv")

    # Perform clustering
    clustered_gdf = perform_hdbscan(data, min_cluster_size=100, min_samples=20, filename='hilo_fall_clusters.geojson', save_output=False)

    # Summarize the clusters
    cluster_summary = summarize_clusters(clustered_gdf)

    # Plot the cluster centroids
    plot_clusters(cluster_summary, "clusters.png")

    #eps = 500
    #min_samples = 20
    #gdf = perform_dbscan(data_cleaned, eps, min_samples)
    #filename = f'hilo_dbscan_eps{eps}_min{min_samples}.png'
    #plot_clusters(gdf, eps, min_samples, filename)
    # Not functioning yet - plots a variety of graphs
    #eps_range= [500, 1000, 2000, 3000]
    #num_batches = 5
    #batch_cluster_and_plot(data_cleaned, eps_range, num_batches)