# Import necessary libaries
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import utm
import hdbscan
import os
import contextily as ctx
import numpy as np
import folium
import webbrowser


def split_into_months(df, city):
    """
    Splits a dataframe into multiple dataframes based on the month extracted from a date column, 
    and saves each month's dataframe as a CSV file in an output directory.

    Parameters:
        df (pandas.DataFrame): The input dataframe that contains a 'date_range_start' column with dates.
        city (str): The name of the city, used in the output file names.

    Side Effects:
        Creates an output directory if it doesn't exist, and saves a CSV file for each month 
        in the specified output directory. Prints the status of each saved file.

    Returns:
        None
    """
    # Extract month from the date
    df['month'] = df['date_range_start'].str[5:7].astype("Int64")

    # Create a directory for saving the output
    output_dir = 'data/foot-traffic/output'
    os.makedirs(output_dir, exist_ok=True)

    # Create and save a dataframe for each month
    for month in range(1, 13):
        month_df = df[df['month'] == month]
        output_file_name = f"{city}_month_{month}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        month_df.to_csv(output_file_path, index=False)
        print(f"Saved month {month} data to {output_file_path}")


def perform_hdbscan(data, min_cluster_size, min_samples, output_file_name=None):
    """
    Perform HDBSCAN clustering on provided geographic data and optionally save the results.

    Parameters:
        data (pd.DataFrame): DataFrame containing the longitude and latitude columns.
        min_cluster_size (int): The minimum size of clusters; not less than this number of points will form a cluster.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. This helps control the degree of noise.
        output_file_name (str): Optional; The name of the file to save the clusters, saved to 'data/foot-traffic/output' if specified.
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame with cluster labels.
        str: The CRS used in the clustering process.
   
    Output:
        GeoJSON file: If an output file name is specified, the clusters are saved to 'data/foot-traffic/output/{output_file_name}'.
    """
    # Clean dataset
    data = data.dropna(subset=['longitude', 'latitude'])

    # Convert DataFrame to GeoDataFrame with longitude and latitude
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data.longitude, data.latitude),
        crs='EPSG:4326'
    )

    # Convert to UTM coordinates for accurate distance measurement
    median_zone_number = int(gdf['geometry'].apply(lambda x: utm.latlon_to_zone_number(x.y, x.x)).median())
    utm_crs = f'+proj=utm +zone={median_zone_number} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
    gdf = gdf.to_crs(utm_crs)

    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(pd.DataFrame({'x': gdf.geometry.x, 'y': gdf.geometry.y}))
    gdf['cluster'] = labels

    # Save output if a file name is specified
    if output_file_name:
        # Define the directory for saving the file
        save_directory = "data/foot-traffic/output"
        os.makedirs(save_directory, exist_ok=True)

        # Full path for saving the file
        save_path = os.path.join(save_directory, output_file_name)

        # Save to GeoJSON
        gdf.to_file(save_path, driver='GeoJSON')
        print(f"Clusters saved to {save_path}")

    return gdf, utm_crs


def summarize_clusters(gdf):
    """
    Summarizes the clusters in a GeoDataFrame that includes a 'cluster' label column.

    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame with a 'cluster' column where -1 indicates noise.
    
    Returns:
        pd.DataFrame: DataFrame where each row represents a cluster with columns for the
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


def plot_cluster_centroids(cluster_summary_df, utm_crs, output_file_name=None):
    """
    Plots the centroids of clusters on a map, annotating each centroid with the number of points in the cluster,
    and overlays this on a basemap for geographical context, with an optional save function.
    
    Parameters:
        cluster_summary_df (DataFrame): DataFrame with columns 'cluster_label', 'num_points', and 'geometry' (centroids).
        utm_crs (str): The CRS used in the cluster summary data.
        output_file_name (str): Optional; The name of the output file, saved to 'data/foot-traffic/plots'.

    Returns:
        folium.Map: A map object displaying the cluster centroids and annotations.
    
    Output:
        HTML file: If an output file name is specified, the map is saved to the specified path as an HTML file.
    """
    # Convert to a GeoDataFrame if necessary
    if not isinstance(cluster_summary_df, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(cluster_summary_df, geometry='geometry', crs=utm_crs)
    else:
        gdf = cluster_summary_df

    # Convert CRS to WGS84 for Folium
    gdf = gdf.to_crs(epsg=4326)

    # Create the base map centered on the first cluster centroid
    if not gdf.empty:
        first_lat = gdf.iloc[0]['geometry'].y
        first_lon = gdf.iloc[0]['geometry'].x
        map_obj = folium.Map(location=[first_lat, first_lon], zoom_start=12)

        # Add centroids and annotations
        for idx, row in gdf.iterrows():
            folium.CircleMarker(
                location=[row['geometry'].y, row['geometry'].x],
                radius=row['num_points'] / 700,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup=f"Cluster {row['cluster_label']}: {row['num_points']} points"
            ).add_to(map_obj)

        # Add a legend to the map
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 160px; 
                    border: 2px solid grey; z-index:9999; font-size:14px;
                    text-align: center; padding-top: 10px;">
                      <b>Legend</b><br>
                      Cluster Centroids<br>
                      <span style="color:blue;">&#11044;</span> Circle markers<br>
                      are proportional<br>
                      in size<br>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

        # Save or display the map
        if output_file_name:
            output_dir = 'data/foot-traffic/plots'
            os.makedirs(output_dir, exist_ok=True)

            output_file_name = output_file_name if output_file_name.endswith('.html') else output_file_name + '.html'
            output_file_path = os.path.join(output_dir, output_file_name)
            map_obj.save(output_file_path)
            print(f"Map saved to {output_file_path}")

            webbrowser.open(f"file://{os.path.abspath(output_file_path)}")
        else:
            temp_file_path = 'temp_map.html'
            map_obj.save(temp_file_path)
            webbrowser.open(f"file://{os.path.abspath(temp_file_path)}")

        return map_obj
    else:
        print("No clusters to display.")
        return None


if __name__ == "__main__":
    main()
