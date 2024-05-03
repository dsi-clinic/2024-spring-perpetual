# Necessary imports
import json
import pandas as pd
import folium
import webbrowser
import requests
import itertools
import polyline
import os


def find_top_businesses_with_related_brands(data, num_top_businesses=10, save=False, city=None):
    """
    Finds the top businesses with related brands and saves them or displays them.

    Parameters:
        data (DataFrame): A dataframe containing foot traffic data.
        num_top_businesses (int): The number of top businesses to consider.
        save (bool): Whether to save the output dataframe.
        city (str): The name of the city to include in the output file name.

    Returns:
        pd.DataFrame: A dataframe of the top businesses with related brands.

    Output:
        CSV file: If saving, the dataframe is saved to 'data/foot-traffic/output/{city}_top_{num_top_businesses}_and_related_brands.csv'.
    """
    # Find the top businesses
    top_visited = data.sort_values(by='raw_visit_counts', ascending=False).drop_duplicates(subset='safegraph_place_id')
    top_visited = top_visited.head(num_top_businesses)

    results = []

    for index, business in top_visited.iterrows():
        # Parse the JSON data in 'related_same_day_brand'
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
                'Safegraph Place ID': business['safegraph_place_id'],  
                'Main Business': business['location_name'],
                'Main Latitude': business['latitude'],
                'Main Longitude': business['longitude'],
                'Related Brand': row['Brand'],
                'Related Brand Latitude': related_brand_info['latitude'],
                'Related Brand Longitude': related_brand_info['longitude'],
                'Related Brand Correlation': row['Count']
            }
            results.append(result)
    
    results_df = pd.DataFrame(results)

    # Save or display the dataframe
    if save:
        output_dir = 'data/foot-traffic/output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the file name incorporating city and number of top businesses
        output_file_name = f"{city}_top_{num_top_businesses}_and_related_brands.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        results_df.to_csv(output_file_path, index=False)
        print(f"Data saved to {output_file_path}")
    else:
        print(results_df)


def generate_base_map_with_points(df, default_location=[37.77, -122.41], default_zoom_start=12):
    """
    Generates a base map centered at a default location and adds points for businesses and related brands from a dataframe.

    Parameters:
        df (DataFrame): A high foot traffic business and related brands dataframe containing 'Safegraph Place ID', 'Main Business', 'Main Latitude', 
                        'Main Longitude', 'Related Brand', 'Related Brand Latitude', 'Related Brand Longitude', and 'Related Brand Correlation'.
        default_location (list): Coordinates to center the base map.
        default_zoom_start (int): The initial zoom level for the map.
    Returns:
        folium.Map: A map object with markers for each business and its related brands.

    Output:
        None
    """
    # Create the base map
    map_obj = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)

    # Add points for businesses and related brands
    for _, row in df.iterrows():
        if pd.notna(row['Main Latitude']) and pd.notna(row['Main Longitude']):
            folium.Marker(
                location=[row['Main Latitude'], row['Main Longitude']],
                popup=f"Main Business: {row['Main Business']}",
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
    Computes the fastest foot routes between main businesses and related brands using Open Source Routing Machine (OSRM).

    Parameters:
        df (DataFrame): A dataframe containing 'Main Business', 'Main Latitude', 'Main Longitude', 'Related Brand', 
                        'Related Brand Latitude', and 'Related Brand Longitude'.
    
   Returns:
        pd.DataFrame: A dataframe of routes, with columns 'Main Business', 'Related Brand', 'Distance', 'Duration', and 'Geometry'.

    Output:
        None
    """
    routes = []
    osrm_url = "http://router.project-osrm.org/route/v1/foot/"

    for index, row in df.iterrows():
        if pd.notna(row['Main Latitude']) and pd.notna(row['Main Longitude']) and pd.notna(row['Related Brand Latitude']) and pd.notna(row['Related Brand Longitude']):
            request_url = f"{osrm_url}{row['Main Longitude']},{row['Main Latitude']};{row['Related Brand Longitude']},{row['Related Brand Latitude']}?overview=full"  # 'full' for complete route geometry
            try:
                response = requests.get(request_url)
                response.raise_for_status()
                route_data = response.json()

                if 'routes' in route_data and route_data['routes']:
                    first_route = route_data['routes'][0]
                    geometry = first_route.get('geometry')  # This is often an encoded polyline
                    decoded_geometry = polyline.decode(geometry)  # Decoding the polyline to a list of (lat, lon) tuples
                    
                    route_info = {
                        'Main Business': row['Main Business'],
                        'Related Brand': row['Related Brand'],
                        'Distance': first_route['distance'],
                        'Duration': first_route['duration'],
                        'Geometry': decoded_geometry  # Store the decoded geometry
                    }
                    routes.append(route_info)
                else:
                    print(f"No route found for {row['Main Business']} to {row['Related Brand']}")
            except requests.RequestException as e:
                print(f"Request failed: {e}")
    
    return pd.DataFrame(routes)


def plot_routes_on_map(df_routes, output_file_name=None):
    """
    Plots routes on a map, with an option to save the map and open it in a web browser.

    Parameters:
        df_routes (DataFrame): A dataframe containing routes information.
        output_file_name (str): Optional; The name of the output file, saved to 'data/foot-traffic/output'.
    
    Returns:
        folium.Map: A map object displaying the routes and annotations.

    Output:
        HTML file: If saving, the map is saved to the specified path.
    """
    # Define a list of colors for different routes
    colors = itertools.cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkblue', 'lightgreen', 'gray', 'black', 'pink'])

    # Create the base map
    if not df_routes.empty:
        first_lat = df_routes.iloc[0]['Geometry'][0][0]
        first_lon = df_routes.iloc[0]['Geometry'][0][1]
        map_obj = folium.Map(location=[first_lat, first_lon], zoom_start=12)

        # Add a legend to the map
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 170px; height: 140px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    ">&nbsp; <b>Legend</b> <br>
                      &nbsp; Main Business &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                      &nbsp; Related Brand &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i><br>
                      &nbsp; Routes weighted <br>
                      &nbsp; by correlation number
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

        # Add the routes to the map
        for index, row in df_routes.iterrows():
            route_color = next(colors)
            popup_text = f"Route from {row['Main Business']} to {row['Related Brand']}"

            # Calculate the opacity based on the related brand correlation integer
            correlation_int = row.get('Related Brand Count', 1)  # Default to 1 if not present
            opacity = min(max(correlation_int / 10, 0.3), 1.0)  # Scale between 0.3 and 1.0

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
                popup=f"Main Business: {row['Main Business']}",
                icon=folium.Icon(color='red')
            ).add_to(map_obj)

            folium.Marker(
                location=[row['Geometry'][-1][0], row['Geometry'][-1][1]],
                popup=f"Related Brand: {row['Related Brand']}",
                icon=folium.Icon(color='blue')
            ).add_to(map_obj)

        # Save the map if specified
        if output_file_name:
            output_dir = 'data/foot-traffic/output'
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
    main()
