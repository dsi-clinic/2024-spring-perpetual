# import packages
import pandas as pd
import folium
import requests
import pandas as pd
import numpy as np
import polyline
import pickle
import configparser
from folium import IFrame
import sys


def initialize_data(filename):
    # read config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    mapbox_token = config['mapbox']['token']

    # read route list
    # filename = 'data/generated_route_list/CAProute_list_20230913_145319.pkl'
    with open(filename, 'rb') as file:
        route_data = pickle.load(file)
    
    return mapbox_token, route_data


def parse_route_data(route_data):
    '''
    route_list is contains 3 routes (corresponding to the number of trucks)
    Each route within route_list is a list of dictionaries
    Each dictionary contains the ObjectID as key and the capacity as value
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

    return route_dict

def parse_supplemental_data(df):

    # create a dictionary from the df dataframe where the key is the index and value is the name
    name_dict = dict(zip(df.index, df['Name']))

    # create a dictionary from the df dataframe where the key is the index and value is the pickup_capacity
    pickup_capacity_dict = dict(zip(df.index, df['Daily_Pickup_Totes']))

    # create a dictionary from the df dataframe where the key is the index and 
    # value is combination of latitude and longitude columns combined as a list
    lat_long_dict = dict(zip(df.index, df[['Latitude', 'Longitude']].values.tolist()))

    return name_dict, pickup_capacity_dict, lat_long_dict

def create_map(route_number, route_dict, name_dict, pickup_capacity_dict, lat_long_dict, mapbox_token):

    # Create a folium map centered at the mean of the locations
    m = folium.Map(location=[29.3013, -94.7977], zoom_start=12)

    starting_url = "https://api.mapbox.com/directions/v5/mapbox/driving/"
    params = {'access_token': mapbox_token, 'overview': 'full'}

    total_duration = 0    
    for route in route_dict[route_number]:
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
            html_content = f"<p>Name: {name}</p><p>pickup: {pickup_capacity}</p>"
            iframe = IFrame(html_content, width=200, height=100)
            popup = folium.Popup(iframe, max_width=2650)


            folium.Marker(location= coordinates, icon=folium.Icon(color= 'red', 
                                                                        prefix="fa"),
                                                                        popup=popup
                        ).add_to(m)

        route = folium.PolyLine(locations=coords, color='red', weight=2)
        route.add_to(m)

        return m


def main():
    filename = sys.argv[1]
    mapbox_token, route_data = initialize_data(filename)
    route_dict = parse_route_data(route_data)
    
    # read supplemental file
    df = pd.read_csv('data/Galveston_data/FUE_capacity_high_adoption_with_outdoors.csv')

    # parse supplemental data
    name_dict, pickup_capacity_dict, lat_long_dict = parse_supplemental_data(df)

    for route_number in route_dict.keys():
        # create map
        m = create_map(route_number, route_dict, name_dict, pickup_capacity_dict, lat_long_dict, mapbox_token)

        # save map
        m.save('data/generated_capacity_map/{}_capacity_map.html'.format(route_number))  


if __name__ == '__main__':
    main()