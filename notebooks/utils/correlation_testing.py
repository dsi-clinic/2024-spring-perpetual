import math
import pandas as pd
from scipy.stats import linregress
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")

def create_df(file_name, city, state):
    lines_left = True
    with open(file_name, encoding='ISO-8859-1') as f:
        header = f.readline()
        whole_df = None
        while True:
            lines = [header]
            for i in range(5000):
                line = f.readline()
                if not line:
                    lines_left = False
                    break
                lines.append(line)
            df = pd.read_csv(io.StringIO("\n".join(lines)))
            df_filtered = df[(df['City'] == city) & (df['State'] == state)]
            if len(df_filtered) > 0:
                whole_df = df_filtered if whole_df is None else pd.concat([whole_df, df_filtered])
            if not lines_left:
                return whole_df
                
def merge(business_df, foot_df, year):
    foot_df['year'] = foot_df['date_range_start'].str[0:4].astype("Int64")
    foot_df_year = foot_df[foot_df['year'] == year]
    foot_df_year['location_name'] = [x.upper() for x in list(foot_df_year['location_name'])]
    foot_df_year['street_address'] = [x.upper() for x in list(foot_df_year['street_address'])]
    foot_df_year_com = foot_df_year.groupby(['location_name', 'latitude', 'longitude', 'street_address']).agg({'raw_visit_counts': 'sum'}).reset_index()
    business_sales = business_df[['Company', 'Address Line 1', 'Latitude', 'Longitude', 'Sales Volume (9) - Location']]
    merged_df = pd.merge(foot_df_year_com, business_sales, 
                         left_on=['street_address'], 
                         right_on=['Address Line 1'], 
                         how='left')

    merged_df = merged_df.dropna()
    
    return merged_df

def calculate_r_value(x, merged_df):
    min_lat = min(list(merged_df['latitude']))
    min_long = min(list(merged_df['longitude']))
    max_lat = max(list(merged_df['latitude']))
    max_long = max(list(merged_df['longitude']))

    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # Find factors of x
    factors = [(i, x // i) for i in range(1, int(math.sqrt(x)) + 1) if x % i == 0]
    # Choose the factors that are closest to each other
    factor1, factor2 = min(factors, key=lambda f: abs(f[0] - f[1]))
    # Calculate latitude and longitude step sizes for x regions
    lat_step = lat_range / factor1
    long_step = long_range / factor2

    # Define a function to assign region based on latitude and longitude
    def assign_region(row):
        lat_region = int((row['latitude'] - min_lat) / lat_step) + 1
        long_region = int((row['longitude'] - min_long) / long_step) + 1
        return (lat_region - 1) * factor2 + long_region

    # Create a new column 'geographic_region' based on latitude and longitude
    merged_df['geographic_region'] = merged_df.apply(assign_region, axis=1)

    # Group by geographic region and aggregate data
    region_aggregate_df = merged_df.groupby('geographic_region').agg({
        'Sales Volume (9) - Location': 'sum',
        'raw_visit_counts': 'sum'
    }).reset_index()

    # Calculate the linear regression
    slope, intercept, r_value, p_value, std_err = linregress(region_aggregate_df['Sales Volume (9) - Location'], region_aggregate_df['raw_visit_counts'])
    return r_value

def r_plot(merged_df):
    # Initialize lists to store x values and r_values
    x_values = list(range(1, 140))
    r_values = []

    # Calculate r_value for each x
    for x in x_values:
        r_value = calculate_r_value(x, merged_df)
        r_values.append(r_value)

    # Plot the results
    plt.plot(x_values, r_values)
    plt.xlabel('Number of Regions (x)')
    plt.ylabel('R Value')
    plt.title('Effect of Region Count on R Value')
    plt.grid(True)
    plt.show(block=True)

def find_addresses(x, merged_df):

    min_lat = min(list(merged_df['latitude']))
    min_long = min(list(merged_df['longitude']))
    max_lat = max(list(merged_df['latitude']))
    max_long = max(list(merged_df['longitude']))

    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # Find factors of x
    factors = [(i, x // i) for i in range(1, int(math.sqrt(x)) + 1) if x % i == 0]

    # Choose the factors that are closest to each other
    factor1, factor2 = min(factors, key=lambda f: abs(f[0] - f[1]))

    # Calculate latitude and longitude ranges
    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # Calculate latitude and longitude step sizes for x regions
    lat_step = lat_range / factor1
    long_step = long_range / factor2

    # Define a function to assign region based on latitude and longitude
    def assign_region(row):
        lat_region = int((row['latitude'] - min_lat) / lat_step) + 1
        long_region = int((row['longitude'] - min_long) / long_step) + 1
        return (lat_region - 1) * factor2 + long_region

    # Create a new column 'geographic_region' based on latitude and longitude
    merged_df['geographic_region'] = merged_df.apply(assign_region, axis=1)

    # Group by geographic region and find top sales in each region
    top_sales_in_regions = merged_df.groupby('geographic_region').apply(lambda group: group.nlargest(1, 'Sales Volume (9) - Location'))

    # Print the street addresses of the businesses with top sales in each region
    for index, row in top_sales_in_regions.iterrows():
        print(f"Region {index}: {row['street_address']}")

def find_top_businesses(business_df, foot_df, year, radius_km=1.0):
    # Filter foot traffic data for the specified year
    foot_df['year'] = foot_df['date_range_start'].str[0:4].astype("Int64")
    foot_df_year = foot_df[foot_df['year'] == year]
    foot_df_year['location_name'] = [x.upper() for x in list(foot_df_year['location_name'])]
    foot_df_year['street_address'] = [x.upper() for x in list(foot_df_year['street_address'])]

    # Group foot traffic data by business location and sum up foot traffic counts
    foot_traffic_by_business = foot_df_year.groupby(['location_name', 'latitude', 'longitude', 'street_address'])['raw_visit_counts'].sum().reset_index()

    top_businesses = []

    for _, business_row in business_df.iterrows():
        # Extract latitude and longitude of the current business
        business_lat = business_row['Latitude']
        business_long = business_row['Longitude']

        # Calculate the distance between the business and foot traffic locations
        distances = cdist([(business_lat, business_long)], foot_traffic_by_business[['latitude', 'longitude']], 
                          metric='euclidean').flatten()

        # Filter foot traffic locations within the specified radius
        foot_traffic_within_radius = foot_traffic_by_business[distances <= radius_km]

        # Calculate total foot traffic in the area surrounding the business
        total_foot_traffic = foot_traffic_within_radius['raw_visit_counts'].sum()

        # Append the total foot traffic along with business information to the list of top businesses
        top_businesses.append({
            'Company': business_row['Company'],
            'Address Line 1': business_row['Address Line 1'],
            'Latitude': business_lat,
            'Longitude': business_long,
            'Total Foot Traffic': total_foot_traffic
        })

    # Convert the list of top businesses to a DataFrame
    top_businesses_df = pd.DataFrame(top_businesses)

    # Sort the DataFrame by total foot traffic in descending order
    top_businesses_df = top_businesses_df.sort_values(by='Total Foot Traffic', ascending=False)
    
    return top_businesses_df

def find_top_unique_business_addresses(business_df, foot_df, year, min_unique_businesses=100, max_radius_km=1.0, radius_step=0.1):
    # Filter foot traffic data for the specified year
    foot_df['year'] = foot_df['date_range_start'].str[0:4].astype("Int64")
    foot_df_year = foot_df[foot_df['year'] == year]
    foot_df_year['location_name'] = [x.upper() for x in list(foot_df_year['location_name'])]
    foot_df_year['street_address'] = [x.upper() for x in list(foot_df_year['street_address'])]

    # Group foot traffic data by business location and sum up foot traffic counts
    foot_traffic_by_business = foot_df_year.groupby(['location_name', 'latitude', 'longitude', 'street_address'])['raw_visit_counts'].sum().reset_index()

    unique_businesses = set()
    radius_km = max_radius_km  # Starting radius

    while radius_km > 0:
        top_businesses = []

        for _, business_row in business_df.iterrows():
            # Extract latitude and longitude of the current business
            business_lat = business_row['Latitude']
            business_long = business_row['Longitude']

            # Calculate the distance between the business and foot traffic locations
            distances = cdist([(business_lat, business_long)], foot_traffic_by_business[['latitude', 'longitude']], 
                              metric='euclidean').flatten()

            # Filter foot traffic locations within the specified radius
            foot_traffic_within_radius = foot_traffic_by_business[distances <= radius_km]

            # Calculate total foot traffic in the area surrounding the business
            total_foot_traffic = foot_traffic_within_radius['raw_visit_counts'].sum()

            # Append the total foot traffic along with business information to the list of top businesses
            top_businesses.append({
                'Company': business_row['Company'],
                'Address Line 1': business_row['Address Line 1'],
                'Latitude': business_lat,
                'Longitude': business_long,
                'Total Foot Traffic': total_foot_traffic
            })

        # Convert the list of top businesses to a DataFrame
        top_businesses_df = pd.DataFrame(top_businesses)

        # Sort the DataFrame by total foot traffic in descending order
        top_businesses_df = top_businesses_df.sort_values(by='Total Foot Traffic', ascending=False)

        # Add unique businesses to the set
        unique_businesses.update(top_businesses_df.drop_duplicates(subset='Total Foot Traffic')['Address Line 1'])

        # Check if enough unique businesses are found
        if len(unique_businesses) >= min_unique_businesses:
            break

        # Increase radius for the next iteration
        radius_km -= radius_step

    # Return the street addresses of the top unique businesses
    return top_businesses_df[top_businesses_df['Address Line 1'].isin(unique_businesses)]['Address Line 1'].tolist()