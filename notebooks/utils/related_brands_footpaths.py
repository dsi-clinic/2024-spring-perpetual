
# Necessary imports
import json
import pandas as pd
import folium
import webbrowser
import os
import requests
import itertools

# Function to create a df for top businesses with highest visit counts and their top 2 related brands
def find_top_businesses_with_related_brands(data):
    # Step 1: Identify the top ~20 businesses by visit counts
    # Now using 'safegraph_place_id' to ensure unique identification of places
    top_visited = data.sort_values(by='raw_visit_counts', ascending=False).drop_duplicates(subset='safegraph_place_id')
    top_visited = top_visited.head(10)
    
    # Initialize a list to hold the result
    results = []

    # Step 2: For each of the top businesses, find the top related brand based on visit counts
    for index, business in top_visited.iterrows():
        # Parse the JSON data in the 'related_same_day_brand' column
        try:
            related_brands = json.loads(business['related_same_day_brand'])
        except json.JSONDecodeError:
            related_brands = {}

        related_brands_df = pd.DataFrame(list(related_brands.items()), columns=['Brand', 'Count'])
        top_related = related_brands_df.sort_values(by='Count', ascending=False).head(1)
        
        for i, row in top_related.iterrows():
            # Find the matching business entry for the related brand using 'location_name'
            related_brand_info = data[data['location_name'] == row['Brand']].iloc[0]
            result = {
                'Safegraph Place ID': business['safegraph_place_id'],  # Include safegraph_place_id
                'Main Business': business['location_name'],
                'Main Latitude': business['latitude'],
                'Main Longitude': business['longitude'],
                'Related Brand': row['Brand'],
                'Related Brand Latitude': related_brand_info['latitude'],
                'Related Brand Longitude': related_brand_info['longitude']
            }
            results.append(result)
    
    return pd.DataFrame(results)


# Create a base map
def generate_base_map(default_location=[37.77, -122.41], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map

# Function to add markers to the map
def add_points_to_map(df, map_obj):
    """Adds markers to the folium map for main businesses and their related brands."""
    for _, row in df.iterrows():
        # Adding markers for main businesses
        if pd.notna(row['Main Latitude']) and pd.notna(row['Main Longitude']):
            folium.Marker(
                location=[row['Main Latitude'], row['Main Longitude']],
                popup=f"Main Business: {row['Main Business']}",
                icon=folium.Icon(color='red')  # Red for main businesses
            ).add_to(map_obj)

        # Check if related brand information exists and add markers
        if pd.notna(row['Related Brand Latitude']) and pd.notna(row['Related Brand Longitude']):
            folium.Marker(
                location=[row['Related Brand Latitude'], row['Related Brand Longitude']],
                popup=f"Related Brand: {row['Related Brand']}",
                icon=folium.Icon(color='blue')  # Blue for related brands
            ).add_to(map_obj)
    
    return map_obj


def compute_fastest_foot_routes(df, travel_mode='foot'):
    routes = []
    osrm_url = f"http://router.project-osrm.org/route/v1/{travel_mode}/"

    for index, row in df.iterrows():
        request_url = f"{osrm_url}{row['Main Longitude']},{row['Main Latitude']};{row['Related Brand Longitude']},{row['Related Brand Latitude']}?overview=full&geometries=geojson"

        try:
            response = requests.get(request_url)
            response.raise_for_status()

            route_data = response.json()
            print("API Response:", route_data)  # Debug: Print API response to check data

            if route_data['routes']:
                first_route = route_data['routes'][0]
                if 'geometry' in first_route:
                    route_info = {
                        'Main Business': row['Main Business'],
                        'Related Brand': row['Related Brand'],
                        'Distance': first_route['distance'],
                        'Duration': first_route['duration'],
                        'Geometry': first_route['geometry']
                    }
                    routes.append(route_info)
                else:
                    print(f"No geometry data for route from {row['Main Business']} to {row['Related Brand']}")
            else:
                print(f"No routes found for {row['Main Business']} to {row['Related Brand']}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")

    return pd.DataFrame(routes)


def plot_routes_on_map(df_routes):
    # Define a list of colors for different routes
    colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'lightgreen', 'gray', 'black', 'pink'])
    
    # Create the base map
    if not df_routes.empty:
        first_lat = df_routes.iloc[0]['Geometry']['coordinates'][0][1]
        first_lon = df_routes.iloc[0]['Geometry']['coordinates'][0][0]
        map = folium.Map(location=[first_lat, first_lon], zoom_start=12)
        
        # Add a legend to the map
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    ">&nbsp; <b>Legend</b> <br>
                      &nbsp; Main Business &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                      &nbsp; Related Brand &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i>
        </div>
        '''
        map.get_root().html.add_child(folium.Element(legend_html))

        # Add the routes to the map
        for index, row in df_routes.iterrows():
            route_color = next(colors)  # Get a color from the cycle
            popup_text = f"Route from {row['Main Business']} to {row['Related Brand']}"
            route_line = folium.PolyLine(
                locations=[(lat, lon) for lon, lat in row['Geometry']['coordinates']],
                weight=5,
                color=route_color,
                popup=folium.Popup(popup_text, parse_html=True)  # Add popup to the route
            ).add_to(map)

            # Add markers for the start (main business) and end (related brand) points
            folium.Marker(
                location=[row['Geometry']['coordinates'][0][1], row['Geometry']['coordinates'][0][0]],
                popup=f"{row['Main Business']}",
                icon=folium.Icon(color='red')
            ).add_to(map)

            folium.Marker(
                location=[row['Geometry']['coordinates'][-1][1], row['Geometry']['coordinates'][-1][0]],
                popup=f"{row['Related Brand']}",
                icon=folium.Icon(color='blue')
            ).add_to(map)

        return map
    else:
        print("No routes to display.")
        return None


if __name__ == "__main__":
    # Load data
    data = pd.read_parquet("data/foot-traffic/hilo_full_patterns.parquet")

    # Find top businesses and their related brands
    top_businesses_related_brands = find_top_businesses_with_related_brands(data)

    # Print the resulting DataFrame to verify correct loading and processing
    print(top_businesses_related_brands)

    # Generate a base map centered around the average location from the data
    #default_location = [top_businesses_related_brands['Main Latitude'].mean(), top_businesses_related_brands['Main Longitude'].mean()]
    #map = generate_base_map(default_location=default_location, default_zoom_start=12)

    # Add points to the map for both main businesses and related brands
    #map = add_points_to_map(top_businesses_related_brands, map)

    # Define the directory to save the map (for just the points)
    #save_directory = "data/foot-traffic/plots"
    #if not os.path.exists(save_directory):
        #os.makedirs(save_directory)  # Create the directory if it does not exist

    # Define the file path (for just the points)
    #map_file_path = os.path.join(save_directory, 'related_brands_hilo.html')

    # Call the function to compute routes
    df_routes = compute_fastest_foot_routes(top_businesses_related_brands)

    # Plot the routes on the map
    map = plot_routes_on_map(df_routes)

    # Define the directory to save the map
    save_directory = "data/foot-traffic/plots"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)  # Create the directory if it does not exist

    # Define the file path
    map_file_path = os.path.join(save_directory, 'hilo_related_brands_footpaths.html')

    # Save the map
    if map is not None:
        map.save(map_file_path)
        # Open the saved map in the default web browser
        webbrowser.open('file://' + os.path.realpath(map_file_path))
    else:
        print("Map could not be generated.")
    
    # Optionally save the map with points (no foot paths)
    #map.save(map_file_path)

    # Automatically open the saved map with points (no foot paths) in the default web browser
    # webbrowser.open('file://' + os.path.realpath(map_file_path))


    
    