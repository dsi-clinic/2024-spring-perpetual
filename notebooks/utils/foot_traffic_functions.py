# import all of the libraries necessary
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import contextily as cx
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
import geopandas as gpd
import pysal
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from matplotlib.patches import Ellipse
from pointpats import centrography
import json

def x_highest_visits(df, x):
    """
    Identifies locations with the highest raw visit counts.
    Args: 
        df - dataframe of foot traffic data (DataFrame)
        x - specified number of locations to be returned (int)
    Returns: list of x number of locations (list of strings)


    """
    return df.sort_values(by='raw_visit_counts', ascending=False)['location_name'].unique()[:x]

def split_into_seasons(df):
    """
    Splits a dataframe into seasons
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: One dataframe for each season (4 dataframes)
    """
    df['month'] = df['date_range_start'].str[5:7].astype("Int64")
    df_winter = df[df['month'] < 4]
    df_spring = df[(df['month'] > 3) & (df['month'] < 7 )]
    df_summer = df[(df['month'] > 6) & (df['month'] < 10 )]
    df_fall = df[df['month'] > 9]
    return df_winter, df_spring, df_summer, df_fall

def morph_and_visualize(df):
    """
    Restructures a given dataset and outputs a visualization of city foot
    traffic.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: Visualization of foot traffic data (plot)
    """
    # Restructure the dataset so each row represents a place visit
    # Make a subset of the dataset
    columns_to_keep = ['placekey', 'location_name', 'latitude', 'longitude', 'raw_visit_counts', 'raw_visitor_counts', 'related_same_day_brand']
    df_subset = df[columns_to_keep].copy()

    # Fill NaNs in the raw_visit_counts column with 0 and make all entries in that column integers
    df_subset['raw_visit_counts'] = pd.to_numeric(df_subset['raw_visit_counts'], errors='coerce').fillna(0).astype(int)

    # Use repeat to expand the DataFrame
    expanded_df = df_subset.loc[df_subset.index.repeat(df_subset['raw_visit_counts'])].reset_index(drop=True)

    df_sorted = df.sort_values(by='raw_visit_counts', ascending=True)
    df_sorted.reset_index(drop=True, inplace=True)
    
    joint_axes = sns.jointplot(x="longitude", y="latitude", hue="raw_visit_counts", data=df_sorted, s=0.5)
    
    cx.add_basemap(
        joint_axes.ax_joint,
        crs="EPSG:4326",
        source=cx.providers.CartoDB.PositronNoLabels
    )

    # Adjust point sizes in the legend
    handles, labels = joint_axes.ax_joint.get_legend_handles_labels()

    # Extract colors from existing legend handles
    legend_colors = [handle.get_color() for handle in handles]

    # Create custom legend handles with both color and size
    legend_handles = [Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color) for color in legend_colors]

    # Add legend with custom handles
    joint_axes.ax_joint.legend(legend_handles, labels, bbox_to_anchor=(1.5, 1.1), title="Visit Counts")

    # Increases size of points in the graph
    joint_axes.ax_joint.collections[0].set_sizes([20])

    plt.show(block=True)

def food_df(df):
    '''
    Filters dataset for only food locations.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: A dataframe with only food locations (1 dataframe)
    '''
    return df[df['top_category'] == 'Restaurants and Other Eating Places']

def create_location_df(dataframe, location_name):
    '''
    Filters dataset for a specific location name.
    Args:
        df - dataframe of foot traffic data (DataFrame)
        location_name - name of a location (string)
    Returns: A dataframe with only locations that match the name given
    (1 dataframe)
    '''
    # Filter for the given location_name
    location_filter = dataframe[dataframe["location_name"] == location_name]
    
    # Select only the relevant columns
    location_df = location_filter[['location_name', 'latitude', 'longitude']].copy()
    
    # Drop rows with NaN values in 'latitude' or 'longitude'
    location_df = location_df.dropna(subset=['latitude', 'longitude'])
    
    # Drop duplicate rows based on 'latitude' and 'longitude' to ensure unique locations
    unique_location_df = location_df.drop_duplicates(subset=['latitude', 'longitude'])
    
    return unique_location_df

def top_x_businesses_and_related(df, x):
    """
    Make visualization with the top x businessed and their corresponding top
    related brands.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: visualization with the top x businessed and their corresponding top
    related brands
    """
    # Create a df with top x most visited businesses and their corresponding locations

    # Create an empty df
    top_visits = pd.DataFrame()

    # Get the names of the top x most visited businesses
    top_business_names = df.sort_values(by="raw_visit_counts", ascending=False)["location_name"].unique()[:x]

    # Temporary list to collect DataFrames
    temp_dfs = []

    for business in top_business_names:
        business_data = df[df["location_name"] == business]
        # Sort and take the first row with the highest "raw_visit_counts"
        top_business_data = business_data.sort_values(by="raw_visit_counts", ascending=False).head(1)
        # Collect the top row for each business
        temp_dfs.append(top_business_data)

    # Concatenate all the top rows into a single DataFrame
    top_visits = pd.concat(temp_dfs)

    # Reset index after concatenating
    top_visits.reset_index(drop=True, inplace=True)

    # Drop any rows with NaN values in 'latitude' or 'longitude'
    top_visits = top_visits.dropna(subset=['latitude', 'longitude'])

    top_related_brands = {}

    for business in top_business_names:
        # Filter for the current business
        business_data = df[df['location_name'] == business].copy()
        
        # Safely convert the JSON string in related_same_day_brand to a list, handling None values
        business_data['related_same_day_brand_list'] = business_data['related_same_day_brand'].apply(
            lambda x: json.loads(x) if x is not None else []
        )
        
        # Explode the DataFrame so each brand has its own row
        all_related_brands = business_data.explode('related_same_day_brand_list')
        
        # Count the most common related same-day brand
        if not all_related_brands.empty:
            top_brand = all_related_brands['related_same_day_brand_list'].value_counts().nlargest(3)
        else:
            top_brand = 'No data'
        
        # Store the result in the dictionary
        top_related_brands[business] = top_brand

    # List of brand DFs
    df_list = []
    
    for _, brand in top_related_brands.items():
        for b in brand.keys():
            df_list.append(create_location_df(df, b))

    top_related_df = pd.concat(df_list, ignore_index=True)

    # Convert df to Geo df
    gdf = gpd.GeoDataFrame(top_visits, geometry=gpd.points_from_xy(top_visits.longitude, top_visits.latitude))
    gdf_related_brands = gpd.GeoDataFrame(top_related_df, geometry=gpd.points_from_xy(top_related_df.longitude, top_related_df.latitude))

    # Set CRS to WGS84
    gdf.crs = "EPSG:4326"
    gdf_related_brands.crs = "EPSG:4326"

    # Convert to Web Mercator for mapping with contextily basemap
    gdf = gdf.to_crs(epsg=3857)
    gdf_related_brands = gdf_related_brands.to_crs(epsg=3857)

    # Basic plot of related same day brand visits and
    # top x highest foot traffic businesses (w/ labels)

    # Set the figure size
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot top locations
    gdf.plot(ax=ax, color='blue', marker='o', label='Top Locations')

    # Plot related brands
    gdf_related_brands.plot(ax=ax, color='red', marker='x', label='Related Brands')

    # Add basemap
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    # Add labels for top locations
    for idx, row in gdf.iterrows():
        name = row['location_name']
        # Check if the name has already been labeled
        ax.text(row.geometry.x, row.geometry.y, s=name, fontsize=8, ha='right', va='bottom', color='blue')

    # Add labels for related brands, ensuring each name is shown only once
    for idx, row in gdf_related_brands.iterrows():
        name = row['location_name']
        # Only label if the name hasn't been labeled yet
        ax.text(row.geometry.x, row.geometry.y, s=name, fontsize=8, ha='left', va='top', color='red')

    ax.set_axis_off()
    plt.legend()
    plt.show(block=True)