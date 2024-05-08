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
import json
import requests

### Set API key
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY')

### GET data and SAVE from RentCast API

import requests
import json

def fetch_all_properties(city, state, property_type, api_key):
    '''
    To handle pagination in API calls, where we might need to make multiple requests to fetch all datam
    we can create a function that dynamically adjusts the offset based on the number of items returned in the response. 
    If a response returns fewer than the maximum limit (500 in your case), 
    it implies there are no more records to fetch.
    '''
    base_url = "https://api.rentcast.io/v1/properties"
    limit = 500
    offset = 0
    all_data = []

    while True:
        # Construct the URL with the current offset
        url = f"{base_url}?city={city}&state={state}&propertyType={property_type}&limit={limit}&offset={offset}"

        # Headers including the API key
        headers = {
            "accept": "application/json",
            "X-Api-Key": api_key
        }

        # Making the API call
        response = requests.get(url, headers=headers)
        data = response.json()

        # Check the number of properties returned
        num_properties = len(data)
        all_data.extend(data)

        # If fewer than limit properties are returned, stop fetching
        if num_properties < limit:
            break
        
        # Otherwise, increase the offset by the limit for the next call
        offset += limit

    # Save the collected data to a JSON file
    with open(f'{city}_{state}_{property_type.replace("-", "_")}_data.json', 'w') as file:
        json.dump(all_data, file, indent=4)

    return f"Data fetched and saved for {city}, {state}, type: {property_type}"

# Optinal function
def fetch_large_residential_buildings(geojson_path, api_key):
    """
    Fetch large residential buildings for a city defined by a GeoJSON file.
    
    Parameters:
    - geojson_path (str): Path to the GeoJSON file defining city boundaries.
    - api_key (str): API key for accessing the RentCast API.
    
    Returns:
    - list: A list of dictionaries, each representing a large residential building.
    """
    # Load GeoJSON file
    gdf = gpd.read_file(geojson_path)
    city_center = gdf.unary_union.centroid
    
    # Initialize variables for API requests
    url_template = "https://api.rentcast.io/v1/properties?latitude={lat}&longitude={lng}&offset={offset}"
    headers = {
        "accept": "application/json",
        "X-Api-Key": api_key
    }
    offset = 0
    large_buildings = []
    max_entries_per_request = 500  # API's limit per request
    
    while True:
        # Prepare URL with current offset
        url = url_template.format(lat=city_center.y, lng=city_center.x, offset=offset)
        
        # Make the API request
        response = requests.get(url, headers=headers)
        response_data = response.json()
        
        # Assuming response_data is a list of properties
        for property in response_data:
            # Example criterion: unitCount > 40, but need to adjust it
            if property.get('features', {}).get('unitCount', 0) > 40:
                large_buildings.append(property)
        
        # Check if we've fetched all available entries
        if len(response_data) < max_entries_per_request:
            break  # Exit loop if fewer than max entries are returned, indicating we've reached the end
        
        offset += max_entries_per_request  # Increase offset for the next request
    
    return large_buildings


### Checking Viability of RentCase API Data


def check_roomCounts(data):
    """
    Prints the formatted address and room count for each property in the data list that contains a room count.
    
    Iterates through a list of property dictionaries, checks for the presence of a room count in the property's features,
    and prints out the property's formatted address alongside the room count if it exists. Additionally, it counts
    the number of properties that contain a room count.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and contains
      a 'formattedAddress' key and a 'features' dictionary that may include a 'roomCount' key.
    
    Returns:
    - int: The number of properties that contain a room count.
    """
    count_contained = 0
    for item in data:
        formattedAddress = item.get('formattedAddress')
        features = item.get('features',{})
        roomCount = features.get('roomCount', None)
        if roomCount is not None:
            print(f"Address: {formattedAddress}, Room Count: {roomCount}")
            count_contained += 1
    return count_contained


def check_unitCounts(data):
    """
    Prints the formatted address and unit count for each property in the data list that contains a unit count.
    
    Iterates through a list of property dictionaries, checks for the presence of a unit count in the property's features,
    and prints out the property's formatted address alongside the unit count if it exists. Additionally, it counts
    the number of properties that contain a unit count.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and contains
      a 'formattedAddress' key and a 'features' dictionary that may include a 'unitCount' key.
    
    Returns:
    - int: The number of properties that contain a unit count.
    """
    count_contained = 0
    for item in data:
        formattedAddress = item.get('formattedAddress')
        features = item.get('features',{})
        unitCount = features.get('unitCount', None)
        if unitCount is not None:
            print(f"Address: {formattedAddress}, Unit Count: {unitCount}")
            count_contained += 1
    return count_contained


def check_features(data):
    """
    Prints the features of each property in the data list that contains additional feature information.
    
    Iterates through a list of property dictionaries and prints the 'features' dictionary for each property,
    if it exists. The 'features' dictionary is assumed to contain additional details about the property.
    
    Parameters:
    - data (list of dict): A list of dictionaries, where each dictionary represents a property and
      may contain a 'features' dictionary with additional property details.
    
    Returns:
    - None
    """
    for item in data:
        features = item.get('features', {})  # Returns {} if 'features' is not found
        if features:
            print(features)


### API Data Cleaning and Exploratory analysis

def filter_properties_by_feature(data, feature_key):
    """
    Filters properties that contain a specific feature.
    """
    filtered = [item for item in data if feature_key in item.get('features', {})]
    return filtered


def print_property_features(data):
    """
    Prints selected features of properties.
    """
    for item in data:
        features = item.get('features', {})
        if features:
            print(f"{item['formattedAddress']}: {features}")


def count_feature_presence(data, feature_key):
    """
    Counts the presence of a specific feature in the property data.
    """
    count = sum(1 for item in data if feature_key in item.get('features', {}))
    return count


def load_and_clean_data(filepath, cols):
    """
    Load JSON data from a file, convert it to a pandas DataFrame, and clean it.
    
    Parameters:
    - filepath (str): The file path to the JSON data file.
    - cols (list): a list of strings of columns that you think are important to drop if they contain NaN values
    
    Returns:
    - pd.DataFrame: A cleaned pandas DataFrame containing the data from the JSON file.
    """
    # Load data from JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data)

    df.dropna(subset=cols, inplace=True)

    return df


def plot_distribution(data, column, title):
    """Plot the distribution of a column."""
    plt.figure(figsize=(12, 6))
    sns.histplot(data[column], kde=True, color='blue', bins=30)
    plt.title(f'Distribution of {title}')
    plt.xlabel(title)
    plt.ylabel('Frequency')
    plt.show()

def plot_correlation(data, x, y):
    """Plot a scatter plot to show correlation and print correlation coefficient."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y, alpha=0.6)
    plt.title(f'Relationship between {x} and {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    correlation = data[x].corr(data[y])
    print(f'Pearson correlation coefficient between {x} and {y}: {correlation:.2f}')


def main(city, gdb_path, api_json_path, parquet_path):
    # Example usage
    geojson_path = 'path_to_your_geojson_file/hilo.geojson'
    api_key = 'your_api_key_here'
    large_residential_buildings = fetch_large_residential_buildings(geojson_path, api_key)

    # For demonstration, print the number of fetched large buildings
    print(f"Number of large residential buildings fetched: {len(large_residential_buildings)}")














