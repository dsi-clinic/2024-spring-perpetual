# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import geopandas as gpd
from pointpats import centrography
import contextily as ctx
import pandas as pd
import argparse
import os


def process_parquet(dataset, output_file_name):
    """
    Restructures the given parquet dataset so that each row represents a place visit.

    Parameters:
        dataset: The original parquet file.
        output_file_name: The desired name for the output CSV file.
    
    Returns:
        None

    Output:
        CSV file: The processed data is saved as a CSV file to 'data/foot-traffic/output/{output_file_name}'.
    """
    columns_to_keep = ['placekey', 'location_name', 'latitude', 'longitude',
                       'raw_visit_counts', 'raw_visitor_counts', 'related_same_day_brand', 'date_range_start']

    # Ensure all necessary columns are present in the dataset
    for col in columns_to_keep:
        if col not in dataset.columns:
            raise ValueError(f"Column '{col}' is missing from the dataset")

    df_subset = dataset[columns_to_keep].copy()
    df_subset['raw_visit_counts'] = pd.to_numeric(df_subset['raw_visit_counts'], errors='coerce').fillna(0).astype(int)
    expanded_df = df_subset.loc[df_subset.index.repeat(df_subset['raw_visit_counts'])].reset_index(drop=True)
    
    # Define the directory path for saving the output
    output_dir = 'data/foot-traffic/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Form the full path for the output file
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Save the processed DataFrame as a CSV
    expanded_df.to_csv(output_file_path, index=False)

    print(f"Processed data has been saved to {output_file_path}")


def plot_hexbin(data_df, gridsize=50, zoom=12, output_file_name=None):
    """
    Create a hexbin plot of geographic data to address point clutter by showing density.

    Parameters:
        data_df (DataFrame): DataFrame containing the geographic data with columns 'longitude' and 'latitude'.
        gridsize (int): The number of hexagons in the x-direction.
        zoom (int): Zoom level for the basemap.
        output_file_name (str): Optional; The name of the output file, saved to 'data/foot-traffic/output' if specified.
    
    Returns:
        None or plt.Figure: Saves the plot to the specified path or displays it directly.
    
    Output:
        PNG file: If an output file name is specified, the plot is saved as a PNG file to the specified path.
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))

    # Generate hexbin map
    hxb = ax.hexbin(
        data_df['longitude'], 
        data_df['latitude'], 
        gridsize=gridsize, 
        cmap='viridis_r',  
        alpha=0.4,         
        linewidths=0
    )

    # Add basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom=zoom)

    # Add a color bar to show the scale
    plt.colorbar(hxb)

    # Remove axis for cleanliness
    ax.set_axis_off()

    # Save the plot if an output file name is specified
    if output_file_name:
        output_dir = 'data/foot-traffic/plots'
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file_path)
        print(f"Plot saved to {output_file_path}")
    else:
        # Show the plot if not saving
        plt.show()


def plot_stdev_ellipse(data_df, output_file_name=None):
    """
    Plot a standard deviational ellipse for geographic data to evaluate dispersion.

    Parameters:
        data_df (DataFrame): A pandas DataFrame containing 'longitude' and 'latitude' columns.
        output_file_name (str): Optional; The name of the output file, saved to 'data/foot-traffic/output' if specified.
    
    Returns:
        None or plt.Figure: Saves the plot to the specified path or displays it directly.
    
    Output:
        PNG file: If an output file name is specified, the plot is saved as a PNG file in the specified directory.
    """
    # Compute the mean and median centers
    mean_center = centrography.mean_center(data_df[["longitude", "latitude"]])
    med_center = centrography.euclidean_median(data_df[["longitude", "latitude"]])

    # Compute the axes and rotation of the ellipse
    major, minor, rotation = centrography.ellipse(data_df[["longitude", "latitude"]])

    # Set up the figure and axis
    f, ax = plt.subplots(1, figsize=(16, 12), dpi=100)

    # Plot the points
    ax.scatter(data_df["longitude"], data_df["latitude"], s=0.5)
    ax.scatter(*mean_center, color="red", marker="x", label="Mean Center")
    ax.scatter(*med_center, color="limegreen", marker="o", label="Median Center")

    # Construct the standard deviation ellipse
    ellipse = Ellipse(
        xy=mean_center, 
        width=major * 2, 
        height=minor * 2,
        angle=np.rad2deg(rotation),
        facecolor="none",
        edgecolor="red",
        linestyle="--",
        label="Standard Deviation Ellipse"
    )
    ax.add_patch(ellipse)

    # Add legend
    ax.legend()

    # Add basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)

    # Save the plot if an output file name is specified
    if output_file_name:
        output_dir = 'data/foot-traffic/plots'
        os.makedirs(output_dir, exist_ok=True)

        output_file_path = os.path.join(output_dir, output_file_name)
        plt.savefig(output_file_path)
        print(f"Plot saved to {output_file_path}")
    else:
        # Show the plot if not saving
        plt.show()


def split_into_months(df, city):
    """
    Takes a city foot traffic dataframe and splits it into 12 dataframes,
    one for each month of the year, saving each dataframe to a file.

    Parameters:
        df (DataFrame): The input dataframe containing 'date_range_start'.
        city (str): The name of the city to use in the output file names.
    
    Returns:
        list: A list of paths to the saved CSV files, one for each month's dataframe.
    
    Output:
        CSV files: One file for each month's dataframe, saved to 'data/foot-traffic/output'.
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


if __name__ == "__main__":
    # Correctly read the Parquet file into a DataFrame
    data = pd.read_parquet('notebooks/hilo_full_patterns.parquet')
    output_file_name = 'processed_hilo_data.csv'
    
    # Call the function with the DataFrame and output file name
    process_parquet(data, output_file_name)