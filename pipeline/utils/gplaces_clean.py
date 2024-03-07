import pandas as pd
import json

# Load JSON data from file
with open('data/poi/hilo_google.json', 'r') as file:
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
    name = hilo_data['displayName']['text']
    categories = ', '.join(hilo_data['types'])
    longitude = hilo_data['location']['longitude']
    latitude = hilo_data['location']['latitude']
    display_address = hilo_data['formattedAddress']

    # Append extracted fields to the respective lists
    ids.append(id)
    names.append(name)
    categories_list.append(categories)
    longitudes.append(longitude)
    latitudes.append(latitude)
    display_addresses_list.append(display_address)

# Create DataFrame
hilo_gplaces_locations = pd.DataFrame({
    'id': ids,
    'name': names,
    'categories': categories_list,
    'longitude': longitudes,
    'latitude': latitudes,
    'display_address': display_addresses_list
})

# Assuming you have already created the DataFrame 'hilo_gplaces_locations'

# Drop duplicates based on 'id'
hilo_gplaces_locations = hilo_gplaces_locations.drop_duplicates(subset=['id'])

# Save DataFrame as CSV
hilo_gplaces_locations.to_csv('data/poi/hilo_gplaces_locations.csv', index=False)