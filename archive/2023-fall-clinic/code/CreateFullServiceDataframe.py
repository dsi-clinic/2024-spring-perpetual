import numpy as np
import pandas as pd

dist_matrix_path = "../data/generated_distance_matrices/distance_matrix_new.npy"
dist_matrix_df_path = "../data/indoor_outdoor_distances_galv.csv"
indoor_pts_path = "../data/galveston_indoor_pts.csv"
outdoor_pts_path = "../data/galveston_outdoor_pts.csv"
old_pts_path = "../data/FUE_Galveston.csv"
all_pts_path = "../data/indoor_outdoor_pts_galv.csv"


def create_full_service_location_df():
    """
    1. Loads the npy distance matrix file
    (made for the galveston moody gardens + indoor + outdoor points)
    and converts it into a csv file --> saves the csv file in the data folder
    2. Concatenates the galveston_indoor_pts and galveston_outdoor_pts dfs
    into one full dataframe will all service locations
    3. Performs data cleaning on the merged service locations df -->
    saves the csv in the data folder

    :params: None
    :return: None
    """
    # load the distance matrix file, it is a numpy array
    dist_matrix_array = np.load(dist_matrix_path)

    # convert the distance matrix to a dataframe
    dist_matrix_df = pd.DataFrame(dist_matrix_array)
    # name columns
    dist_matrix_df.columns = [str(i) for i in range(dist_matrix_array.shape[1])]

    # save to csv
    dist_matrix_df.to_csv(dist_matrix_df_path, index=False)

    # Load the indoor pts df and the original FUE df
    indoor = pd.read_csv(indoor_pts_path)
    old_galveston = pd.read_csv(old_pts_path)

    # 1. merge the indoor pts df with the columns "Weekly_Dropoff_Totes"
    # and "Daily_Pickup_Totes" from the original df
    galveston_sub = old_galveston.loc[
        :, ["Name", "Weekly_Dropoff_Totes", "Daily_Pickup_Totes"]
    ]
    indoor = pd.merge(indoor, galveston_sub, on="Name", how="left")

    # 2. add the "Moody Gardens" location to the top of the indoor dataframe
    # extract the Moody Gardens columns from the old dataframe
    moody_gardens = old_galveston.loc[
        (old_galveston.loc[:, "Name"] == "Moody Gardens"),
        (
            [
                "Name",
                "Longitude",
                "Latitude",
                "Daily_Pickup_Totes",
                "Weekly_Dropoff_Totes",
            ]
        ),
    ]

    # concatenate the column on top of the indoor points df
    moody_plus_indoor = pd.concat([moody_gardens, indoor])

    # 3. add a column "location_type" = "indoor"
    # to every point in the indoor dataframe
    moody_plus_indoor.loc[:, "location_type"] = "indoor"

    # 4. add a column "pickup_type" = "Truck"
    # to every point in the indoor dataframe
    moody_plus_indoor.loc[:, "pickup_type"] = "Truck"

    # Load outdoor pts df
    outdoor = pd.read_csv(outdoor_pts_path)

    # 5. add a daily pickup value of 1.0 and weekly dropoff value
    # to 0.0 to every outdoor point
    outdoor.loc[:, "Daily_Pickup_Totes"] = 1.0
    outdoor.loc[:, "Weekly_Dropoff_Totes"] = 0.0

    # 6. add a column "location_type" = "outdoor"
    # to every point in the outdoor dataframe
    outdoor.loc[:, "location_type"] = "outdoor"

    # 7. capitalize the "longitude" and "latitude" columns in the outdoor df
    outdoor = outdoor.rename(
        columns={"longitude": "Longitude", "latitude": "Latitude"}
    )

    # 8. concatonate the indoor and outdoor points into a single dataframe
    full_service_locations = pd.concat([moody_plus_indoor, outdoor])

    # 9. sets "location_type" = "depot" for "Moody Garden"
    full_service_locations.loc[
        (full_service_locations.loc[:, "Name"] == "Moody Gardens"),
        "location_type",
    ] = "depot"

    # 10. fill in missing values for pickup and dropoff totes
    full_service_locations.loc[
        full_service_locations.loc[:, "Daily_Pickup_Totes"].isna(),
        "Daily_Pickup_Totes",
    ] = 1.0
    full_service_locations.loc[
        full_service_locations.loc[:, "Weekly_Dropoff_Totes"].isna(),
        "Weekly_Dropoff_Totes",
    ] = 1.0

    columns_to_convert = ["Daily_Pickup_Totes", "Weekly_Dropoff_Totes"]
    full_service_locations[columns_to_convert] = full_service_locations[
        columns_to_convert
    ].astype(int)
    # reset index and check
    full_service_locations.reset_index(inplace=True, drop=True)

    # save to csv file
    full_service_locations.to_csv(all_pts_path, index=False)


if __name__ == "__main__":
    create_full_service_location_df()
