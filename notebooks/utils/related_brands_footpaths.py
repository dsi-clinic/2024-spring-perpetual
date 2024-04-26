
# Necessary imports
import json
import pandas as pd
import folium
import webbrowser
import os
import requests
import itertools
import polyline


# Function to create a df for top businesses with highest visit counts and their top related brands
def find_top_businesses_with_related_brands(data):
    top_visited = data.sort_values(by='raw_visit_counts', ascending=False).drop_duplicates(subset='safegraph_place_id')
    top_visited = top_visited.head(10)
    
    # Initialize a list to hold the result
    results = []

    # For each of the top 10 businesses, find the top related brand based on visit counts
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
                'Safegraph Place ID': business['safegraph_place_id'],  
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
    return folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)


# Add points to map
def add_points_to_map(df, map_obj):
    # Adding markers for main businesses
    for _, row in df.iterrows():
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


# Compute fastest foot routes
def compute_fastest_foot_routes(df):
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


def plot_routes_on_map(df_routes):
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
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    ">&nbsp; <b>Legend</b> <br>
                      &nbsp; Main Business &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                      &nbsp; Related Brand &nbsp; <i class="fa fa-map-marker fa-2x" style="color:blue"></i>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))

        # Add the routes to the map
        for index, row in df_routes.iterrows():
            route_color = next(colors) 
            popup_text = f"Route from {row['Main Business']} to {row['Related Brand']}"
            route_line = folium.PolyLine(
                locations=row['Geometry'],
                weight=5,
                color=route_color,
                popup=folium.Popup(popup_text, parse_html=True)  # Add popup to the route
            ).add_to(map_obj)

            # Add markers for the start (main business) and end (related brand) points
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

        return map_obj
    else:
        print("No routes to display.")
        return None


if __name__ == "__main__":
    # Example usage of the functions
    # Load your dataset
    data = pd.read_parquet("data/foot-traffic/hilo_full_patterns.parquet")

    # Find top businesses and related brands
    top_businesses_related_brands = find_top_businesses_with_related_brands(data)

    # Generate a base map centered on the average location of the main businesses
    if not top_businesses_related_brands.empty:
        map_obj = generate_base_map([top_businesses_related_brands['Main Latitude'].mean(), top_businesses_related_brands['Main Longitude'].mean()])
        add_points_to_map(top_businesses_related_brands, map_obj)
        map_obj.save("map.html")

    # Compute fastest routes
    routes = compute_fastest_foot_routes(top_businesses_related_brands)
    
    # Define the path where the map should be saved
    save_path = "data/foot-traffic/plots/routes_map.html"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot routes on map and save to HTML
    if not routes.empty:
        route_map = plot_routes_on_map(routes)
        if route_map is not None:
            full_path = os.path.abspath(save_path)  # Get the absolute path of the file
            route_map.save(full_path)  # Save the map as an HTML file
            print(f"Route map saved successfully. You can view it by opening this file in a web browser: {full_path}")
        else:
            print("Failed to create routes map.")
