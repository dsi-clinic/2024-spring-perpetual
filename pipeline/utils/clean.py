#file used for cleaning scraped data
import pandas as pd


filepath = 'data/poi/hilo_yelp_original.json'

yelp_scrape_dataframe = pd.read_json(filepath)



csv_filepath = 'data/poi/yelp_data.csv'

# Save DataFrame as CSV
yelp_scrape_dataframe.to_csv(csv_filepath, index=False)