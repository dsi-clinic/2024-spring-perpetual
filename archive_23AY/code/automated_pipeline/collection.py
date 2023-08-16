from config import *
from functions import *
import requests
import json
import time
import numpy as np
import pandas as pd
import re
import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect Business Data")
    parser.add_argument("city", help="Name of City")
    #parser.add_argument("fue_file", help="path to output FUE CSV")
    #parser.add_argument("all_business_file", help="path to output all business CSV")
    args = parser.parse_args()
    
    # access coordinates in config.py
    small_circles = coords[args.city]
    small_radius = radius[args.city]
    center_cord = center_coords[args.city]
    center_rad = center_radius[args.city]

    # list of common and uncommon places
    common = places["common"]
    uncommon = places["uncommon"]
    clct_common = places["clct_common"]
    clct_uncommon = places["clct_uncommon"]
    clct_residential = places["clct_residential"]
    all_places = common+uncommon

    # retrieve keys
    google_key = keys["google"]
    yelp_key = keys["yelp"]

    #google process
    df_uncommon = run_scrape(center_cord, center_rad, uncommon, google_key)
    df_common = run_scrape(small_circles, small_radius, common, google_key)
    df_clct_common = run_scrape(small_circles, small_radius, clct_common, google_key)
    df_clct_uncommon = run_scrape(center_cord, center_rad, clct_uncommon, google_key)
    df_clct_residential = run_scrape(center_cord,center_rad, clct_residential, google_key)


    google = clean_data(df_dirty, args.city)

    #yelp process
    yelp_dirty = run_yelp_scrape(full_name[args.city], all_places, yelp_key)

    yelp = clean_yelp_data(yelp_dirty, args.city)

    #merge and cleaning
    lat_lng(df_common)
    lat_lng(df_uncommon)
    lat_lng(df_clct_common)
    lat_lng(df_clct_uncommon)
    lat_lng(df_clct_residential)

    df_dirty = pd.concat([df_uncommon, df_common, df_clct_common, df_clct_uncommon, df_clct_residential, yelp])

    aa_merged = remove_standard_duplicates(df_dirty)
    aa_merged = remove_fuzzy_duplicates(aa_merged)

    # save to csv
    aa_merged.to_csv(args.fue_file)


    


   
