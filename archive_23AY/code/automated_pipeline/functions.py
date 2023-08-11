import requests
import json
import time
import numpy as np
import pandas as pd
import re
from fuzzywuzzy import process

def get_nearby_places(lat_lng, radius, business_type, api_key):
    # define the endpoint URL
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # define the parameters for the API request
    params = {
        "location": lat_lng,
        "radius": radius,
        "region": "us",
        "type": business_type,
        "key": api_key
    }

    # send the API request and get the response
    response = requests.get(url, params=params)
    data = response.json()

    # create a list to hold the details for each place
    places = []

    # iterate through each place in the results
    for place in data["results"]:
        # extract the details we want
        name = place["name"]
        lat = place["geometry"]["location"]["lat"]
        lng = place["geometry"]["location"]["lng"]
        place_id = place["place_id"]
        rating = place.get("rating", None)
        business_status = place.get("business_status", None)
        address = place.get("vicinity", None)
        addressf = place.get("formatted_address", None)
        types = place.get("types", None)


        # clean up the address string
        if address:
            address = re.sub(r"<.*?>", "", address)
            address = address.strip()

        # add the details to the list
        places.append({
            "name": name,
            "lat": lat,
            "lng": lng,
            "place_id": place_id,
            "rating": rating,
            "business_status": business_status,
            "address": address,
            "addressf" : addressf
            "types" : types
        })
        while "next_page_token" in data['results']:
            params['pagetoken'] = data['results']['next_page_token']
            res = requests.get(url, params = params)
            results =  json.loads(res.content)
            df = df.append(results['results'], ignore_index=True)
            time.sleep(1)

    # create a DataFrame from the list of places
    df = pd.DataFrame(places).assign(place_type=business_type)
    
    return df

def run_scrape(coords, radii, businesses, key):
    # coords: coordinates of circles
    # radii: radii of circles
    # businesses: types of businesses we are interested in
    # key: google API key
    df = pd.DataFrame()
    for i in range(len(coords)): #for each set of coords/radius
        for j in businesses: #for each business type
            if df.shape[0]==0: 
                df = get_nearby_places(coords[i], radii[i], j, key)
            else:
                df = pd.concat([df, get_nearby_places(coords[i], radii[i], j, key)])
    return df

def clean_data_step1(df, city):
    # drop duplicate rows based on the "address" column
    df = df.drop_duplicates(subset=["place_id"], inplace=True)
    # filter the DataFrame to only include rows where the "address" column includes the city name
    df = df[df["address"].str.contains(city, case=False)]
    df['category'] = df['place_type'].apply(lambda x: x[0])
    return df



def save(df, city):
    df.to_csv("../data/"+city+"_businesses.csv")



def get_yelp_data(api_key, location, business_type):
    url = 'https://api.yelp.com/v3/businesses/search'
    headers = {'Authorization': 'Bearer %s' % api_key}
    
    # Set the initial values
    limit = 50
    offset = 0
    all_results = []
    
    # Loop through all the pages
    while True:
        # Set the parameters for the API request
        params = {
            'location': location,
            'term': business_type,
            'limit': limit,
            'offset': offset
        }
        
        # Make the API request
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        businesses = data['businesses']
        
        # Add the businesses to the list of all results
        all_results += businesses
        
        # Increment the offset
        offset += limit
        
        # If there are no more businesses, break out of the loop
        if len(businesses) < limit:
            break
            
    # Create a dataframe from the results
    df = pd.DataFrame(all_results)
    
    # Extract the necessary columns
    df = df[['name', 'location', 'coordinates', 'id', 'rating', 'review_count']]
    
    # Clean up the data
    df['address'] = df['location'].apply(lambda x: ', '.join(x['display_address']))
    df.drop('location', axis=1, inplace=True)
    
    # Reorder the columns
    df = df[['name', 'address', 'coordinates', 'id', 'rating', 'review_count']]
    
    return df

def run_yelp_scrape(place, businesses, key):
    df = pd.DataFrame()
    for i in businesses: #for each business type
        if df.shape[0]==0: 
            df = get_yelp_data(key, place, i)
        else:
            df = pd.concat([df, get_yelp_data(key, place, i)])
    return df

# Define a function to help as detect NaN value
def isNaN(num):
    return num != num


def clean_yelp_data(df, city):
    # drop duplicate rows based on the "address" column
    df = df.drop_duplicates(subset=["id"], in)
    # filter the DataFrame to only include rows where the "address" column includes the city name
    df = df[df["address"].str.contains(city, case=False)]
    df = df.iloc[:, :-3]
    # Clean datasets. The "hours" column was messy and not in a string form, so I extracted information from the column and transited it to
    # string form. Similar for the "transactions" column
    city_open = []
    city_hour_type = []
    city_is_open_now = []
    city_transactions = []
    for i in range(df.shape[0]):
        if not isNaN(df["hours"].iloc[i]):
            city_open.append(df["hours"].iloc[i][0]["open"])
            city_hour_type.append(df["hours"].iloc[i][0]["hours_type"])
            city_is_open_now.append(df["hours"].iloc[i][0]["is_open_now"])
        else:
            city_open.append("")
            city_hour_type.append("")
            city_is_open_now.append("")
        if (df["transactions"].iloc[i] != []) and (not isNaN(df["transactions"].iloc[i])):
            city_transactions.append(df["transactions"].iloc[i][0])
        else:
            city_transactions.append("")

    # I stored the info I extracted from "hours" and "transactions" form to new columns of the dataset
    df["open"] = city_open
    df["hour_type"] = city_hour_type
    df["is_open_now"] = city_is_open_now
    df["transactions"] = city_transactions
    df.reset_index()
    for i in range(len(df)):
    lst = eval(df['categories'][i])
    type_list = []
    for j in range(len(lst)):
        type_list.append(lst[j]['alias'])
    df['categories'][i] = type_list
    df = df.rename(columns={
    'name': 'Name',
    'coordinates.latitude': 'Latitude',
    'coordinates.longitude': 'Longitude',
    'categories': 'category',
    'full_address': 'Address',
    'is_closed': 'business_status',
    'review_count': 'user_ratings_total',
    'price': 'price_level'
    })

    return df


# define all the neccessary functions in the data cleaning and merging process

def merge_dataframes(*args):
    concat = pd.concat(args)
    
    return concat

def remove_standard_duplicates(concat):
    merged = concat.drop_duplicates(subset=['Name', 'Address'])
    merged = merged.drop(columns=['Unnamed: 0'])
    #merged = merged[[merged.columns[0], merged.columns[1], merged.columns[2]]]
    
    return merged


def remove_fuzzy_duplicates(merged):
    merged['Name + Address'] = merged['Name'] + merged['Address']
    merged = merged.astype(str)
    unique_strings = list(merged['Name + Address'].unique())
    merged['second_best_match'] = merged['Name + Address'].apply(lambda x: process.extract(x, unique_strings, limit=2)[1][0])
    merged['second_best_score'] = merged['Name + Address'].apply(lambda x: process.extract(x, unique_strings, limit=2)[1][1])
    potential_matches = merged.loc[merged['second_best_score'] >= 92]
    potential_matches = potential_matches.sort_values('Name')
    potential_matches = potential_matches.drop(potential_matches.index[::2])
    complete = merged.loc[merged['second_best_score'] < 92]
    complete = complete.append(potential_matches)
    
    return complete

# creating two new columns for latitude and longitude
def lat_lng(df):
    lat = []
    long = []
    for i in range(len(df)):
        if isNaN(df['Coordinates'][i]):
            df['Coordinates'][i] = "{'lat': None, 'lng': None}"
        df['Coordinates'][i] = eval(df['Coordinates'][i])
        try:
            lat.append(df['Coordinates'][i]['lat'])
            long.append(df['Coordinates'][i]['lng'])
        except:
            lat.append(df['Coordinates'][i]['latitude'])
            long.append(df['Coordinates'][i]['longitude'])
    df['Latitude'] = lat
    df['Longitude'] = long