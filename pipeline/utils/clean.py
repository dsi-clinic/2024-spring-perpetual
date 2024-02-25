import pandas as pd
import json

# Load JSON data from file
with open('data/poi/hilo_yelp_original.json', 'r') as file:
    hilo_data_list = json.load(file)

# Initialize lists to store extracted fields
ids = []
names = []
categories_list = []
longitudes = []
latitudes = []
display_addresses_list = []

# Iterate over each JSON object in the list
for hilo_data in hilo_data_list:
    # Extract fields from the current JSON object
    id = hilo_data['id']
    name = hilo_data['name']
    categories = ', '.join([category['title'] for category in hilo_data['categories']])
    longitude = hilo_data['coordinates']['longitude']
    latitude = hilo_data['coordinates']['latitude']
    display_address = ', '.join(hilo_data['location']['display_address'])
    
    # Append extracted fields to the respective lists
    ids.append(id)
    names.append(name)
    categories_list.append(categories)
    longitudes.append(longitude)
    latitudes.append(latitude)
    display_addresses_list.append(display_address)

# Create DataFrame
hilo_yelp_locations = pd.DataFrame({
    'id': ids,
    'name': names,
    'categories': categories_list,
    'longitude': longitudes,
    'latitude': latitudes,
    'display_address': display_addresses_list
})

# Assuming you have already created the DataFrame 'hilo_yelp_locations'

# Save DataFrame as CSV
hilo_yelp_locations.to_csv('data/poi/hilo_yelp_locations.csv', index=False)

