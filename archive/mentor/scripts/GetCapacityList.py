import datetime
import os
import pickle
import sys

import pandas as pd


def get_data(data_dir):
    """
    Fetch the data from a CSV file.

    :param data_dir: Path to the data directory.
    :return: DataFrame.
    :raises FileNotFoundError: If the specified file is not found.
    """

    # Take file name from the command line
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        raise ValueError("File name must be provided.")

    csv_path = os.path.join(data_dir, file_name)

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' was not found.")

    return df


def save_capacity_list(df, data_dir):
    """
    Generate a list of capacities from the dataframe and save it.

    :param df: Dataframe
    :param data_dir: Path to the data directory.
    """
    # Get timestamp string
    current_datetime = datetime.datetime.now()
    timestamp_str = current_datetime.strftime("%Y-%m-%d_%H:%M")

    # save list as pkl file
    capacity_file = (
        f"{data_dir}/capacity_lists/capacity_list_{timestamp_str}.pkl"
    )
    with open(capacity_file, "wb") as f:
        pickle.dump(list(map(int, df["Daily_Pickup_Totes"].tolist())), f)


def main():

    # Construct the path to the required directories
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "..", "data")

    # Initialize the dataframe
    df = get_data(data_dir)

    # Save the capacity list to a file
    save_capacity_list(df, data_dir)


if __name__ == "__main__":
    main()
