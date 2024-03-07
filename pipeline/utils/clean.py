import pandas as pd
import json


# Function to load JSON data from file and convert it to a DataFrame
def json_to_dataframe(json_path):
    with open(json_path, "r") as file:
        data_list = json.load(file)
    return pd.DataFrame(data_list)


# Load JSON data from files
gplaces_json_path = "data/poi/hilo_google.json"
yelp_json_path = "data/poi/hilo_yelp.json"

# Applying the function to convert JSON to DateFrame
gplaces_df = json_to_dataframe(gplaces_json_path)
yelp_df = json_to_dataframe(yelp_json_path)

# Combine the DataFrames
combined_df = pd.concat([gplaces_df, yelp_df])

# Save the combined DataFrame to a CSV file
combined_csv_path = "data/poi/hilo_google_yelp_combined.csv"
combined_df.to_csv(combined_csv_path, index=False)
