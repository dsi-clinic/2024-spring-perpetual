import ast
import configparser

import pandas as pd


def generate_single_source_of_truth():
    """
    Generate a single source of truth DataFrame by
    merging indoor and outdoor data and the source location

    Reads data from indoor and outdoor CSV files specified
    in the configuration file, updates and prepares the data,
    and saves the resulting DataFrame as a CSV file.
    Save the location of the new Dataframe in configuration file.

    :return: None
    """

    # Initialize the parser
    config = configparser.ConfigParser()
    config.read("../utils/config_inputs.ini")

    # read config file
    # take file name from config
    file_name_indoor = config["original data source"]["indoor"]
    file_name_outdoor = config["original data source"]["outdoor"]
    file_name_old = config["original data source"]["old"]

    # read indoor and outdoor data file
    df_indoor = pd.read_csv(file_name_indoor)
    df_outdoor = pd.read_csv(file_name_outdoor)
    old_galveston = pd.read_csv(file_name_old)

    # update location_type and pickup_type for each dataset
    df_indoor.loc[:, "location_type"] = "indoor"
    df_indoor.loc[:, "pickup_type"] = "truck"

    df_outdoor.loc[:, "Daily_Pickup_Totes"] = 1.0
    df_outdoor.loc[:, "Weekly_Dropoff_Totes"] = 0.0
    df_outdoor.loc[:, "location_type"] = "outdoor"

    # prepare data for analyze
    df_outdoor = df_outdoor.rename(
        columns={"longitude": "Longitude", "latitude": "Latitude"}
    )

    # add tote number info
    # when the indoor and oudoor df don't have pickup/dropoff totes number
    if "Weekly_Dropoff_Totes" not in df_indoor.columns:
        galveston_sub = old_galveston.loc[
            :, ["Name", "Weekly_Dropoff_Totes", "Daily_Pickup_Totes"]
        ]
        df_indoor = pd.merge(df_indoor, galveston_sub, on="Name", how="left")

    # take source location from config
    source_location = ast.literal_eval(config["original data source"]["source"])
    source_location_lon = source_location[0]
    source_location_lat = source_location[1]
    df_source = pd.DataFrame(
        {"Longitude": [source_location_lon], "Latitude": [source_location_lat]}
    )

    # concat indoor points, outdoor points and source location into one df
    single_source_truth = pd.concat([df_source, df_indoor, df_outdoor])

    # save single_source to a file
    filename = "../data/archive/single_source_of_truth.csv"
    single_source_truth.to_csv(filename, index=False)

    # add the path to single_source_of_truth to config
    # if "original data source" not in config:
    #     config["original data source"] = {}
    # config["original data source"]["single_source_of_truth"] = filename
    # with open("../utils/config_inputs.ini", "w") as configfile:
    #     config.write(configfile)

    print(f"single source of truth generated under the file {filename}")


if __name__ == "__main__":
    generate_single_source_of_truth()
