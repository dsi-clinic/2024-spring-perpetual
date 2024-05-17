# Necessary imports
import json
import pandas as pd
import folium
import webbrowser
import requests
import itertools
import polyline
import os
import numpy as np


def find_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth using their latitude and longitude.

    Parameters:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
        float: The distance between the two points in kilometers.

    Output:
        None
    """
    R = 6371.0  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def find_top_businesses_with_related_brands(df, num_top_businesses=10, save=False, city=None):
    """
    Finds the top foot traffic businesses and their highest correlated related brand and saves them in a CSV or DataFrame.

    Parameters:
        df (DataFrame): A dataframe containing foot traffic data.
        num_top_businesses (int): The number of top businesses to consider.
        save (bool): Whether to save the output dataframe.
        city (str): The name of the city to include in the output file name.

    Returns:
        pd.DataFrame: A dataframe of the top foot traffic businesses with related brands information.

    Output:
        CSV file: If saving, the dataframe is saved to 'data/foot-traffic/output/{city}_top_{num_top_businesses}_and_related_brands.csv'.
    """
    # Find the top businesses
    top_visited = df.sort_values(by='raw_visit_counts', ascending=False).drop_duplicates(subset='safegraph_place_id')
    top_visited = top_visited.head(num_top_businesses)

    results = []

    for index, business in top_visited.iterrows():
        # Parse the JSON data in 'related_same_day_brand'
        try:
            related_brands = json.loads(business['related_same_day_brand'])
        except json.JSONDecodeError:
            related_brands = {}

        # Get highest correlation same day brand per high foot traffic business
        related_brands_df = pd.DataFrame(list(related_brands.items()), columns=['Brand', 'Count'])
        top_related = related_brands_df.sort_values(by='Count', ascending=False).head(1)

        for i, row in top_related.iterrows():
            # Find all locations of the related brand
            related_brand_locations = df[df['location_name'] == row['Brand']]

           
            related_brand_locations = related_brand_locations.copy()
            
            # Calculate the distance to each related brand location
            related_brand_locations.loc[:, 'distance'] = related_brand_locations.apply(
                lambda x: find_haversine_distance(business['latitude'], business['longitude'], x['latitude'], x['longitude']), axis=1)
            
            # Find the nearest related brand location
            nearest_related_brand = related_brand_locations.loc[related_brand_locations['distance'].idxmin()]

            result = {
                'Safegraph Place ID': business['safegraph_place_id'],  
                'High Traffic Location': business['location_name'],
                'High Traffic Latitude': business['latitude'],
                'High Traffic Longitude': business['longitude'],
                'Related Brand': row['Brand'],
                'Related Brand Latitude': nearest_related_brand['latitude'],
                'Related Brand Longitude': nearest_related_brand['longitude'],
                'Related Brand Correlation': row['Count'],
                'Distance to Related Brand (km)': nearest_related_brand['distance']
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)

    # Save or display the dataframe
    if save:
        output_dir = 'data/foot-traffic/output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the file name using city and number of top businesses
        output_file_name = f"{city}_top_{num_top_businesses}_and_related_brands.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        results_df.to_csv(output_file_path, index=False)
        print(f"Data saved to {output_file_path}")
    
    return results_df

def generate_base_map_with_points(df, default_location=[19.7071, -155.0885], default_zoom_start=12):
    """
    Generates a base map with markers for high traffic locations and each one's top related brand.

    Parameters:
        df (DataFrame): A dataframe of the top foot traffic businesses with related brands information.

    Returns:
        folium.Map: A Folium Map object with markers added for high traffic locations and related brands.

    Output: 
        None
        
    """
    # Create the base map
    map_obj = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    # Add points for businesses and related brands
    for _, row in df.iterrows():
        if pd.notna(row['High Traffic Latitude']) and pd.notna(row['High Traffic Longitude']):
            folium.Marker(
                location=[row['High Traffic Latitude'], row['High Traffic Longitude']],
                popup=f"High Traffic Location: {row['High Traffic Location']}",
                icon=folium.Icon(color='red')
            ).add_to(map_obj)

        if pd.notna(row['Related Brand Latitude']) and pd.notna(row['Related Brand Longitude']):
            folium.Marker(
                location=[row['Related Brand Latitude'], row['Related Brand Longitude']],
                popup=f"Related Brand: {row['Related Brand']}",
                icon=folium.Icon(color='blue')
            ).add_to(map_obj)

    return map_obj


def compute_fastest_foot_routes(df):
    """
    Computes the fastest foot routes between high foot traffic businesses and related brands using Open Source Routing Machine (OSRM) and stores them in a DataFrame.

    Parameters:
        df (DataFrame): A dataframe of the top foot traffic businesses with related brands information.
    
   Returns:
        pd.DataFrame: A dataframe of fastest foot routes, with columns 'High Traffic Business', 'Related Brand', 'Related Brand Correlation', 'Distance', 'Duration', and 'Geometry'.

    Output:
        None
    """
    routes = []
    osrm_url = "http://router.project-osrm.org/route/v1/foot/"

    for index, row in df.iterrows():
        if pd.notna(row['High Traffic Latitude']) and pd.notna(row['High Traffic Longitude']) and pd.notna(row['Related Brand Latitude']) and pd.notna(row['Related Brand Longitude']):
            request_url = f"{osrm_url}{row['High Traffic Longitude']},{row['High Traffic Latitude']};{row['Related Brand Longitude']},{row['Related Brand Latitude']}?overview=full"
            try:
                response = requests.get(request_url)
                response.raise_for_status()
                
                route_data = response.json()

                if 'routes' in route_data and route_data['routes']:
                    first_route = route_data['routes'][0]
                    geometry = first_route.get('geometry')
                    decoded_geometry = polyline.decode(geometry)
                    
                    route_info = {
                        'High Traffic Location': row['High Traffic Location'],
                        'Related Brand': row['Related Brand'],
                        'Related Brand Correlation': row['Related Brand Correlation'],
                        'Distance': first_route['distance'],
                        'Duration': first_route['duration'],
                        'Geometry': decoded_geometry 
                    }
                    routes.append(route_info)
                else:
                    print(f"No route found for {row['High Traffic Location']} to {row['Related Brand']}")
            except requests.RequestException as e:
                print(f"Request failed: {e}")
    
    return pd.DataFrame(routes)


def plot_routes_on_map(df_routes, output_file_name=None):
    """
    Plots routes on a map, with an option to save the map and open it in a web browser.

    Parameters:
        df_routes (DataFrame): A dataframe containing fastest foot routes information.
        output_file_name (str): Optional; The name of the output file, saved to 'data/foot-traffic/output'.
    
    Returns:
        folium.Map: A map object displaying the routes and annotations.

    Output:
        HTML file: If saving, the map is saved to the specified path.
    """
    # Define a list of colors for different routes
    colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'lightgreen', 'gray', 'black', 'pink'])

    # Determine the range of correlation values
    min_corr = df_routes['Related Brand Correlation'].min()
    max_corr = df_routes['Related Brand Correlation'].max()

    # Function to compute opacity based on correlation
    def compute_opacity(correlation, min_corr, max_corr):
        """
        Compute the opacity for the route based on the correlation value.
        Normalize the correlation value to a range of 0.3 to 1.0.
        """
        return 0.5 + (correlation - min_corr) / (max_corr - min_corr) * (1.0 - 0.5)

    # Create the base map
    if not df_routes.empty:
        first_lat = df_routes.iloc[0]['Geometry'][0][0]
        first_lon = df_routes.iloc[0]['Geometry'][0][1]
        map_obj = folium.Map(location=[first_lat, first_lon], zoom_start=12)

        # Add a legend to the map
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 170px; height: 110px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    ">&nbsp; <b>Legend</b> <br>
                      &nbsp; High Traffic Location &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                      &nbsp; Related Brand &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i><br>
                      &nbsp; Routes weighted <br>
                      &nbsp; by correlation number
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

        # Add the routes to the map
        for index, row in df_routes.iterrows():
            route_color = next(colors)
            popup_text = f"Route from {row['High Traffic Location']} to {row['Related Brand']}"

            # Calculate the opacity based on the related brand correlation integer
            correlation_int = row['Related Brand Correlation']
            opacity = compute_opacity(correlation_int, min_corr, max_corr)

            folium.PolyLine(
                locations=row['Geometry'],
                weight=5,
                color=route_color,
                opacity=opacity,
                popup=folium.Popup(popup_text, parse_html=True)
            ).add_to(map_obj)

            # Add markers for the start and end points
            folium.Marker(
                location=[row['Geometry'][0][0], row['Geometry'][0][1]],
                popup=f"High Traffic Location: {row['High Traffic Location']}",
                icon=folium.Icon(color='red')
            ).add_to(map_obj)

            folium.Marker(
                location=[row['Geometry'][-1][0], row['Geometry'][-1][1]],
                popup=f"Related Brand: {row['Related Brand']}",
                icon=folium.Icon(color='blue')
            ).add_to(map_obj)

        # Save the map if specified
        if output_file_name:
            output_dir = 'data/foot-traffic/plots'
            os.makedirs(output_dir, exist_ok=True)
            
            output_file_name = output_file_name if output_file_name.endswith('.html') else output_file_name + '.html'
            output_file_path = os.path.join(output_dir, output_file_name)
            map_obj.save(output_file_path)
            print(f"Map saved to {output_file_path}")

            # Open in a web browser
            webbrowser.open(f"file://{os.path.abspath(output_file_path)}")

        else:
            # Directly display the map in a web browser
            temp_file_path = 'temp_map.html'
            map_obj.save(temp_file_path)
            webbrowser.open(f"file://{os.path.abspath(temp_file_path)}")
        
        return map_obj
    else:
        print("No routes to display.")
        return None


if __name__ == "__main__":
    
    # Name the city for output map
    city = "hilo"

    # Load data from the Parquet file
    df = pd.read_parquet("notebooks/hilo_full_patterns.parquet")

    # Find the top businesses with related brands
    top_businesses_with_related_brands = find_top_businesses_with_related_brands(df)
    
    # Compute the fastest foot routes
    fastest_routes = compute_fastest_foot_routes(top_businesses_with_related_brands)
    
    # Plot the routes on a map
    map_obj = plot_routes_on_map(fastest_routes, output_file_name=f"{city}_routes_map.html")
