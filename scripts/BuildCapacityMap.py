# import libraries
import pandas as pd
import numpy as np
import folium
import requests
import pandas as pd
import numpy as np
import polyline
import pickle
import configparser
import sys
import os
from folium import IFrame

# import the BeautifyIcon class
import folium.plugins


def read_config(root_dir):
    """
    Initialize the config file

    :param root_dir: Path to the root directory.
    :return: Mapbox token
    """
    # Initialize the parser
    config = configparser.ConfigParser()
    # Read the config file
    config_path = os.path.join(root_dir, 'config.ini')    
    config.read(config_path)
    mapbox_token = config['mapbox']['token']
    import ast
    central_coordinates = ast.literal_eval(config['mapping']['map_center'])

    return mapbox_token, central_coordinates

def fetch_data(filename, data_dir):
    '''
    :param filename: name of the file to be read
    :param data_dir: path to the data directory
    :return: route_data: list of dictionaries
    '''

    filename = os.path.join(data_dir, filename)
    with open(filename, 'rb') as file:
        route_data = pickle.load(file)
    
    return route_data


def parse_route_data(route_data):
    '''

    :param route_data: list of dictionaries
    :return: route_dict: dictionary of route name as key and route list as value
    '''

    # convert route_list to a dictionary
    # The code below just create a dictionary with route name as key and route list as value
    # convert route_data to a dictionary
    route_dict = {}
    for i in range(len(route_data)):
        route_dict['Route {}'.format(i+1)] = route_data[i]


    # For each key in route_dict, if the list size is greater than 25, split the list into lists of 25
    for key in route_dict.keys():
        if len(route_dict[key]) > 25:
            route_dict[key] = [route_dict[key][i:i+25] for i in range(0, len(route_dict[key]), 25)]
        else:
            route_dict[key] = [route_dict[key]]

    return route_dict

def parse_supplemental_data(datafile):

    df = pd.read_csv(datafile)

    # create a dictionary from the df dataframe where the key is the index and value is the name
    name_dict = dict(zip(df.index, df['Name']))

    # create a dictionary from the df dataframe where the key is the index and value is the pickup_capacity
    pickup_capacity_dict = dict(zip(df.index, df['Daily_Pickup_Totes']))

    # create a dictionary from the df dataframe where the key is the index and value is the category
    category_dict = dict(zip(df.index, df['location_type']))

    # create a dictionary from the df dataframe where the key is the index and 
    # value is combination of latitude and longitude columns combined as a list
    lat_long_dict = dict(zip(df.index, df[['Latitude', 'Longitude']].values.tolist()))

    return name_dict, category_dict, pickup_capacity_dict, lat_long_dict

def create_map(route_dict, name_dict, category_dict, pickup_capacity_dict, lat_long_dict, mapbox_token, central_coordinates):
    '''
    :param route_dict: dictionary of route name as key and route list as value
    :param name_dict: dictionary of index as key and name as value
    :param category_dict: dictionary of index as key and category as value
    :param pickup_capacity_dict: dictionary of index as key and pickup_capacity as value
    :param lat_long_dict: dictionary of index as key and latitude and longitude as value
    :param mapbox_token: mapbox token
    :param central_coordinates: central coordinates of the map
    :return: folium map
    '''

    starting_url = "https://api.mapbox.com/directions/v5/mapbox/driving/"
    params = {'access_token': mapbox_token, 'overview': 'full'}

    # Create a folium map centered at the mean of the locations
    m = folium.Map(location=central_coordinates, zoom_start=12)

    total_duration = 0   
    # create a list of 6 colors
    color_lst = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']
    for truck_route in route_dict.keys():
        route_color = color_lst.pop()

        for route in route_dict[truck_route]:
            coordinate_lst = []
            for i in route:
                stop_index = list(i.keys())[0]
                accumulated_load = list(i.values())[0]
                coordinates = (lat_long_dict[stop_index][1], lat_long_dict[stop_index][0])
                coordinate_lst.append(coordinates)
            end_url = ";".join([f"{lon},{lat}" for lon, lat in coordinate_lst])
            url = starting_url + end_url
            response = requests.get(url, params=params)
            res = response.json()
            total_duration += res['routes'][0]['duration']

            polyline_str = res['routes'][0]['geometry']
            coords = polyline.decode(polyline_str)

            for i in route:
                stop_index = list(i.keys())[0]
                coordinates = [lat_long_dict[stop_index][0], lat_long_dict[stop_index][1]]
                name = name_dict[stop_index]
                pickup_capacity = pickup_capacity_dict[stop_index]
                category = category_dict[stop_index]
                html_content = f"<p>Name: {name}</p><p>Category: {category}</p>\
                <p>Pickup Totes: {pickup_capacity}</p> \
                <p>Route Number: {truck_route}</p>"
                iframe = IFrame(html_content, width=200, height=100)
                popup = folium.Popup(iframe, max_width=2650)

                if category == 'outdoor':
                    icon_dot = folium.plugins.BeautifyIcon(
                        border_color="#000300", icon_shape="circle-dot", iconSize=(10,10),
                        backgroundColor=route_color, border_width=1
                    )

                    folium.Marker(location=coordinates, 
                        popup=popup, icon=icon_dot).add_to(m)
                elif category == 'depot':
                    folium.Marker(
                        coordinates,
                        popup=popup,
                        icon=folium.Icon(icon='star')
                    ).add_to(m)
                else:
                    icon_dot = folium.plugins.BeautifyIcon(
                        border_color="#000300", icon_shape="rectangle-dot", iconSize=(10,10),
                        backgroundColor=route_color, border_width=1
                    )

                    folium.Marker(location=coordinates, 
                        popup=popup, icon=icon_dot).add_to(m)


            route = folium.PolyLine(locations=coords, color=route_color, weight=1)
            route.add_to(m)
    
    html_path = 'code/legend.html'
    with open(html_path, 'r') as file:
        legend_html = file.read()
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def get_file_paths():
    """
    Retrieves file paths from command line arguments.

    :return: A tuple containing the matrix file path and the capacity file path.
    :raises IndexError: If the required arguments are not provided.
    """
    if len(sys.argv) < 3:
        raise IndexError("Not enough arguments provided. Please provide a matrix file and a capacity file.")

    data_file = sys.argv[1]
    route_list = sys.argv[2]

    return data_file, route_list


def main():
    """Entry point of the program."""

    # Construct the path to the required directories
    current_dir = os.path.dirname(__file__)
    root_dir = os.path.join(current_dir, '..')
    data_dir = os.path.join(root_dir, 'data')

    # Get file paths
    data_file, route_list = get_file_paths(data_dir)

    # Get routing configuration
    try:
        mapbox_token, central_coordinates = read_config(root_dir)
    except FileNotFoundError:
        raise FileNotFoundError("Config file not found.")

    route_data = fetch_data(route_list)
    route_dict = parse_route_data(route_data)

    # parse supplemental data
    name_dict, category_dict, pickup_capacity_dict, \
        lat_long_dict = parse_supplemental_data(data_file)

    map = create_map(route_dict, name_dict, category_dict, \
                   pickup_capacity_dict, lat_long_dict, mapbox_token, central_coordinates)

    # save map
    map.save('capacity_map.html')  

if __name__ == '__main__':
    main()