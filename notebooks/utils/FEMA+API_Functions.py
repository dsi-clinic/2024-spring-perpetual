import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import fiona
import pyarrow.parquet as pq
from matplotlib.patches import Patch
from pyproj import Proj, Transformer
import requests
import geopandas as gpd
from dotenv import load_dotenv
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

### Load data from various sources

def load_data(gdb_path, api_json_path, parquet_path):
    """
    Load spatial and tabular data from various sources into GeoDataFrames and DataFrames.

    Parameters:
    - gdb_path (str): The file path to the Geodatabase containing geographic data.
    - api_json_path (str): The file path to the JSON file containing API data with longitude and latitude.
    - parquet_path (str): The file path to the Parquet file containing foot traffic data.

    Returns:
    - tuple: A tuple containing three GeoDataFrames:
        1. gdf: GeoDataFrame loaded from the Geodatabase.
        2. gdf_api: GeoDataFrame created from the API JSON data.
        3. geo_parquet_df: GeoDataFrame created from the Parquet file foot traffic data.
    """
    layers = fiona.listlayers(gdb_path)
    gdf = gpd.read_file(gdb_path, layer=layers[0])
    api_data = pd.read_json(api_json_path)
    api_data['geometry'] = api_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf_api = gpd.GeoDataFrame(api_data, geometry='geometry')
    gdf_api.set_crs('EPSG:4326', inplace=True)
    foot_traffic_data = pd.read_parquet(parquet_path)
    foot_traffic_data['geometry'] = foot_traffic_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    geo_parquet_df = gpd.GeoDataFrame(foot_traffic_data, geometry='geometry')
    return gdf, gdf_api, geo_parquet_df

### Data Pre-processing

def convert_to_geodataframe(data, lon_col='longitude', lat_col='latitude', crs='EPSG:4326'):
    """
    Converts a list of dictionaries to a GeoDataFrame.

    Parameters:
    - data (list of dict): Data to convert, each dictionary represents a property.
    - lon_col (str): Column name for longitude values.
    - lat_col (str): Column name for latitude values.
    - crs (str): Coordinate reference system to use for the GeoDataFrame.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the data with a geometry column.
    """
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs)
    return gdf


def perform_spatial_join(gdf1, gdf2, how='inner', op='intersects'):
    """
    Performs a spatial join between two GeoDataFrames.

    Parameters:
    - gdf1 (gpd.GeoDataFrame): The first GeoDataFrame.
    - gdf2 (gpd.GeoDataFrame): The second GeoDataFrame to join with the first.
    - how (str): Type of join, 'left', 'right', 'inner' (default).
    - op (str): Spatial operation to use, 'intersects' (default), 'contains', etc.

    Returns:
    - gpd.GeoDataFrame: The result of the spatial join.
    """
    return gpd.sjoin(gdf1, gdf2, how=how, op=op)



def plot_geodataframes(gdfs, colors, labels):
    """
    Plots multiple GeoDataFrames.

    Parameters:
    - gdfs (list of gpd.GeoDataFrame): List of GeoDataFrames to plot.
    - colors (list of str): Colors for each GeoDataFrame.
    - labels (list of str): Labels for each GeoDataFrame in the legend.

    Displays:
    - A plot with all GeoDataFrames visualized.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for gdf, color, label in zip(gdfs, colors, labels):
        gdf.plot(ax=ax, color=color, label=label)
    plt.legend()
    plt.show()

# Data Analysis and Visualization

def analyze_large_apartments(joined):
    """
    Identify large apartments from the joined GeoDataFrame based on square footage thresholds.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing joined data from multiple sources.

    Returns:
    - GeoDataFrame: A filtered GeoDataFrame containing only entries that meet the large apartment criteria.
    
    Outputs:
    - Prints the number of units meeting the large apartment criteria.
    """
    large_apartment_threshold = 3000
    joined = joined.dropna(subset=['squareFootage', 'SQFEET'])
    print(f"We currently have {len(joined)} units in the joined dataset, after excluding those without squareFootage and SQFEET data")
    large_apartments_gdf = joined[joined['squareFootage'] >= large_apartment_threshold]
    large_apartments_gdf = large_apartments_gdf[large_apartments_gdf['SQFEET'] >= large_apartment_threshold]
    return large_apartments_gdf

def plot_large_apartments(joined, large_apartments_gdf):
    """
    Plot a map showing large apartments over the geographic context of building footprints.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing all building data.
    - large_apartments_gdf (GeoDataFrame): The GeoDataFrame containing only large apartments.

    Outputs:
    - A plot showing building footprints and highlighting large apartments.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    joined.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5, label='Building Footprints')
    large_apartments_gdf.plot(ax=ax, color='cyan', markersize=10, label='Large Apartments')
    building_patch = Patch(color='lightgrey', label='Building Footprints')
    apartment_patch = Patch(color='cyan', label='Large Apartments')
    ax.legend(handles=[building_patch, apartment_patch])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Large Residential Buildings Within Building Footprints')
    ax.set_axis_off()
    plt.show()

def analyze_architecture(joined):
    """
    Analyze and plot the architecture types from the features data within a GeoDataFrame.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing joined data with a 'features' column.

    Outputs:
    - A plot showing the distribution of architecture types based on the API data.
    """
    joined_drop_features = joined.dropna(subset=['features'])
    joined_drop_features['architectureType'] = joined_drop_features['features'].apply(lambda x: x.get('architectureType', None))
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    joined_drop_features.plot(column='architectureType', ax=ax, legend=True, alpha=0.32)
    leg = ax.get_legend()
    leg.set_bbox_to_anchor((1, 0.5))
    leg.set_title('Architecture Type')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Building Footprints by Architecture Type (Based on API data)')
    ax.set_axis_off()
    plt.show()

def analyze_square_footage_correlation(joined):
    """
    Analyze and visualize the correlation between square footage data from API and FEMA sources.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing square footage data from multiple sources.

    Outputs:
    - A scatter plot showing the relationship between square footages from different sources.
    - Prints the Pearson correlation coefficient between the two sources of square footage data.
    """
    joined = joined.dropna(subset=['squareFootage', 'SQFEET', 'SQMETERS', 'features'])
    sns.scatterplot(data=joined, x='squareFootage', y='SQFEET', alpha=0.6)
    plt.title('Relationship between different sources of Square Footage')
    plt.xlabel('Square Footage (From API)')
    plt.ylabel('Square Footage (From FEMA)')
    plt.show()
    correlation_sq = joined['squareFootage'].corr(joined['SQFEET'])
    print(f'Pearson correlation coefficient between API Square Feet and that of FEMA: {correlation_sq:.2f}')


def calculate_units(row):
    """
    Calculate the estimated number of units in a building based on its square footage and property type.

    Parameters:
    - row (pd.Series): A pandas Series representing a row of a DataFrame, expected to contain
                       the building's property type and its total square footage.

    Returns:
    - float or None: The estimated number of units in the building if the property type is recognized
                     and square footage is available. Returns None if the property type is not recognized
                     or square footage is missing.

    The function uses predefined average unit areas for different property types to estimate the number of units.
    If the property type is not in the predefined list or the square footage is not provided, the function returns None.
    """
    average_unit_areas = {
        'Single Family': 2299,
        'Multifamily': 1046,
        'Condo': 592,
        'Apartment': 592,
        'Townhouse': 592,
        'Manufactured': 2000,
        'Land': 2000
    }
    building_type = row.get('propertyType')
    average_area = average_unit_areas.get(building_type, None)
    if average_area is not None and 'SQFEET' in row and row['SQFEET'] is not None:
        return row['SQFEET'] / average_area
    else:
        return None
    
def analyze_units(joined):
    # Drop entries without necessary data
    joined = joined.dropna(subset=['squareFootage', 'SQFEET', 'SQMETERS', 'PRIM_OCC'])

    # Apply the calculate_units function
    joined['number_of_units'] = joined.apply(calculate_units, axis=1)

    # Count entries with more than 40 units
    number_of_entries_over_40_units = (joined['number_of_units'] > 40).sum()
    print(f"We see that after filtering, out of {len(joined)}, we have {number_of_entries_over_40_units} buildings that exceed 40 units")


def process_and_join_foot_traffic(joined, foot_traffic_data):
    """
    Prepare foot traffic data and perform a spatial join with joined GeoJSON data.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing the joined building data.
    - foot_traffic_data (DataFrame): The DataFrame containing foot traffic data with longitude and latitude.

    Returns:
    - GeoDataFrame: The result of the spatial join.
    """
    # Define projections
    proj_wgs84 = Proj(proj='latlong', datum='WGS84')
    proj_utm = Proj(proj="utm", zone=5, datum='WGS84')  # Adjust zone based on your location

    # Create a transformer
    transformer_to_utm = Transformer.from_proj(proj_wgs84, proj_utm, always_xy=True)
    transformer_to_wgs84 = Transformer.from_proj(proj_utm, proj_wgs84, always_xy=True)

    # Convert foot traffic data to GeoDataFrame
    foot_traffic_data['geometry'] = foot_traffic_data.apply(
        lambda row: Point(row['longitude'], row['latitude']), axis=1)
    geo_parquet_df = gpd.GeoDataFrame(foot_traffic_data, geometry='geometry')
    geo_parquet_df.set_crs(joined.crs, allow_override=True)

    # Buffer the points
    def buffer_in_meters(lon, lat, meters=10):
        x, y = transformer_to_utm.transform(lon, lat)
        point_utm = Point(x, y)
        buffered_point_utm = point_utm.buffer(meters)
        if buffered_point_utm.is_empty:
            return None
        exterior_coords = [(x, y) for x, y in zip(*buffered_point_utm.exterior.coords.xy)]
        transformed_coords = [transformer_to_wgs84.transform(x, y) for x, y in exterior_coords]
        return Polygon(transformed_coords)

    geo_parquet_df['geometry'] = geo_parquet_df.apply(
        lambda row: buffer_in_meters(row['longitude'], row['latitude']), axis=1)

    # Perform spatial join
    intersect_foot_geo = gpd.sjoin(joined, geo_parquet_df, how="left", op='intersects')
    print("Number of intersections found after buffering and cleaning:", len(intersect_foot_geo))

    return intersect_foot_geo


def prepare_and_visualize_data(intersect_foot_geo):
    """
    Analyze the estimated number of units in each building and identify those with more than 40 units.

    Parameters:
    - joined (GeoDataFrame): The GeoDataFrame containing joined data which includes property types and square footage.

    Outputs:
    - Modifies the 'joined' GeoDataFrame by adding a 'number_of_units' column, which contains the calculated number of units for each property based on its square footage and property type.
    - Prints the number of buildings that exceed 40 units after filtering out entries lacking necessary data.

    Note:
    - This function relies on the 'calculate_units' function to compute the number of units per building, which must be defined and correctly handle the types of buildings present in 'joined'.
    """
    # Convert columns to numeric and handle missing values
    cols_to_analyze = ['raw_visit_counts', 'raw_visitor_counts', 'SQFEET', 'number_of_units', 'lastSalePrice']
    intersect_foot_geo[cols_to_analyze] = intersect_foot_geo[cols_to_analyze].apply(pd.to_numeric, errors='coerce')
    
    # Check for and handle duplicate indexes
    if intersect_foot_geo.index.duplicated().any():
        print("Duplicate indexes found!")
        print(intersect_foot_geo[intersect_foot_geo.index.duplicated(keep=False)])
    intersect_foot_geo = intersect_foot_geo.reset_index(drop=True)
    intersect_foot_geo = intersect_foot_geo[~intersect_foot_geo.index.duplicated(keep='first')]
    
    # Generate pair plots for selected columns
    sns.pairplot(intersect_foot_geo.dropna(subset=cols_to_analyze), vars=cols_to_analyze[:-1])
    plt.show()

    # Drop NaN values for 'lastSalePrice' and generate pair plots
    intersect_foot_geo.dropna(subset=['lastSalePrice'], inplace=True)
    sns.pairplot(intersect_foot_geo, vars=cols_to_analyze)
    plt.show()

    # Generate a correlation heatmap
    correlation_matrix = intersect_foot_geo[cols_to_analyze].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

    return intersect_foot_geo


def main(city, gdb_path, api_json_path, parquet_path):
    gdf, gdf_api, foot_traffic_data = load_data(gdb_path, api_json_path, parquet_path)
    joined = gpd.sjoin(gdf_api, gdf, how="inner", op='intersects')
    large_apartments_gdf = analyze_large_apartments(joined)
    plot_large_apartments(joined, large_apartments_gdf)
    analyze_architecture(joined)
    analyze_square_footage_correlation(joined)
    analyze_units(joined)
    intersect_foot_geo = process_and_join_foot_traffic(joined, foot_traffic_data)  # Process and join foot traffic data
    prepared_data = prepare_and_visualize_data(intersect_foot_geo)  # Prepare data and visualize


