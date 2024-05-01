# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import geopandas as gpd
from pointpats import centrography
import contextily as ctx
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def process_parquet(dataset):
    """
    Restructures the given parquet datset so that each row represents a place visit.
    
    Parameters:
        dataset: The original parquet file.
        
    Returns:
        : A new dataframe (as a CSV file) where each row represents a single visit.
    """
    columns_to_keep = ['placekey', 'location_name', 'latitude', 'longitude', 
                       'raw_visit_counts', 'raw_visitor_counts', 'related_same_day_brand', 'date_range_start']
    df_subset = dataset[columns_to_keep].copy()
    df_subset['raw_visit_counts'] = pd.to_numeric(df_subset['raw_visit_counts'], errors='coerce').fillna(0).astype(int)
    df_subset = df_subset.reset_index(drop=True)
    expanded_df = df_subset.loc[df_subset.index.repeat(df_subset['raw_visit_counts'])].reset_index(drop=True)
    
    return expanded_df
# add save path
# specify file name

def plot_hexbin(data_df, gridsize=50, zoom=12):
    """
    Create a hexbin plot of geographic data to address point clutter by showing density.

    Parameters:
        data_df (DataFrame): DataFrame containing the geographic data with columns 'longitude' and 'latitude'.
        gridsize (int): The number of hexagons in the x-direction.
        zoom (int): Zoom level for the basemap.
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(16, 12))

    # Generate hexbin map
    hxb = ax.hexbin(
        data_df['longitude'], 
        data_df['latitude'], 
        gridsize=gridsize, 
        cmap='viridis_r',  # Using a reversed Viridis colormap
        alpha=0.4,         # Slightly transparent hexbins to see the map beneath
        linewidths=0
    )

    # Add basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, zoom=zoom)

    # Add a color bar to show the scale
    plt.colorbar(hxb)

    # Remove axis for cleanliness
    ax.set_axis_off()

    # Show the plot
    plt.show()

# Example command line input:
# plot_hexbin(dataset, gridsize=50, zoom=12)
# add optional save path
# specify file name


def plot_stdev_ellipse(data_df, show_plot=True, save_plot=False, filename='ellipse_plot.png'):
    """
    Plot a standard deviational ellipse for geographic data to evaluate dispersion.
    
    Parameters:
        data_df (DataFrame): A pandas DataFrame containing 'longitude' and 'latitude' columns.
        show_plot (bool): If True, display the plot window.
        save_plot (bool): If True, save the plot to a file.
        filename (str): The filename to save the plot to if save_plot is True.
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

    # Display or save the plot based on the function parameters
    if save_plot:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()

# Example command line input:
# plot_stdev_ellipse(path_to_dataset, show_plot=True, save_plot=True, filename='ellipse_plot.png')
# add save path
# specify file name


def split_into_months(df):
    """
    Takes a city foot traffic dataframe and splits it into 12 dataframes,
    one for each month of the year.
    """
    df['month'] = df['date_range_start'].str[5:7].astype("Int64")  # Extract month from the date
    months = {}
    for month in range(1, 13):
        months[month] = df[df['month'] == month]
    return months
# add save path
# specify file name

if __name__ == "__main__":
    main()