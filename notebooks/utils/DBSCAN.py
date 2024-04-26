# Import necessary libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import utm
import hdbscan
import os
import contextily as ctx


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
    #data = data.sample(n=100000, random_state=42)  # Sample 100,000 points randomly

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

    return gdf, utm_crs


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

    if clusters.empty:
        return pd.DataFrame(columns=['cluster_label', 'num_points', 'geometry'])

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


def plot_cluster_centroids(cluster_summary_df, utm_crs):
    """
    Plots the centroids of clusters on a map, annotating each centroid with the number of points in the cluster,
    and overlays this on a basemap for geographical context.
    
    Parameters:
    - cluster_summary_df (pd.DataFrame): DataFrame with columns 'cluster_label', 'num_points', and 'geometry' (centroids).
    """
    
    # Assuming the centroid geometries are in latitude and longitude
    gdf = gpd.GeoDataFrame(cluster_summary_df, geometry='geometry', crs=utm_crs)

    # Check and handle NaNs or infinite values
    if gdf.isna().any().any():
        gdf.dropna(inplace=True)  # Drop NaNs

    # Convert CRS to EPSG:3857 for contextily
    gdf = gdf.to_crs(epsg=3857)

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Adjust marker size based on the number of points, ensuring visibility
    marker_sizes = gdf['num_points'] / gdf['num_points'].max() * 100  # Scale marker size

    # Plot each centroid
    gdf.plot(ax=ax, color='blue', markersize=marker_sizes, alpha=0.6)

    # Annotate the centroid with the number of points in the cluster
    for idx, row in gdf.iterrows():
        ax.text(row.geometry.x, row.geometry.y, f'{row["num_points"]}', fontsize=8, ha='left')

    # Add a basemap with adjusted zoom if needed
    try:
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
    except ValueError:
        # Handle potential ValueError if zoom level is too high
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron, zoom=12)  # A more reasonable, fixed zoom level

    # Set plot parameters
    ax.set_title('Cluster Centroids with Number of Points')
    ax.axis('off') 

    plt.show()


if __name__ == "__main__":
    main()