import requests
import geopandas as gpd

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

# # Example usage
# geojson_path = 'path_to_your_geojson_file/hilo.geojson'
# api_key = 'your_api_key_here'
# large_residential_buildings = fetch_large_residential_buildings(geojson_path, api_key)

# # For demonstration, print the number of fetched large buildings
# print(f"Number of large residential buildings fetched: {len(large_residential_buildings)}")