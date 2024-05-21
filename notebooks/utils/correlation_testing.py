import io
import math
import warnings

import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import linregress

warnings.filterwarnings("ignore")


def create_df(file_name, city, state):
    """
    Create a business dataframe based on business sales
    data from a particular year
    Args:
        file_name - name of the file (str)
        city - name of the city (str)
        state - abbreviation for state (str)
    Returns: business dataframe based on business sales data (DataFrame)
    """
    lines_left = True
    with open(file_name, encoding="ISO-8859-1") as f:
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
            df_filtered = df[(df["City"] == city) & (df["State"] == state)]
            if len(df_filtered) > 0:
                whole_df = (
                    df_filtered
                    if whole_df is None
                    else pd.concat([whole_df, df_filtered])
                )
            if not lines_left:
                return whole_df


def filter_year(foot_df, year):
    """
    Filters foot traffic dataset to a particular year
    Args:
        foot_df - foot traffic dataframe (DataFrame)
    Returns: filtered foot traffic dataset (DataFrame)
    """
    foot_df["year"] = foot_df["date_range_start"].str[0:4].astype("Int64")
    foot_df_year = foot_df[foot_df["year"] == year]
    foot_df_year["location_name"] = [
        x.upper() for x in list(foot_df_year["location_name"])
    ]
    foot_df_year["street_address"] = [
        x.upper() for x in list(foot_df_year["street_address"])
    ]

    return foot_df_year


def merge(business_df, foot_df, year):
    """
    Merge business dataframe and foot traffic dataframe for a particular year
    Args:
        business_df - business dataframe (DataFrame)
        foot_df - foot traffic dataframe (DataFrame)
        year - the year (int)
    Returns: Merged dataframe (DataFrame)
    """
    foot_df_year = filter_year(foot_df, year)
    foot_df_year_com = (
        foot_df_year.groupby(
            ["location_name", "latitude", "longitude", "street_address"]
        )
        .agg({"raw_visit_counts": "sum"})
        .reset_index()
    )
    business_sales = business_df[
        [
            "Company",
            "Address Line 1",
            "Latitude",
            "Longitude",
            "Sales Volume (9) - Location",
        ]
    ]
    merged_df = pd.merge(
        foot_df_year_com,
        business_sales,
        left_on=["street_address"],
        right_on=["Address Line 1"],
        how="left",
    )

    merged_df = merged_df.dropna()

    return merged_df


def calculate_r_value(x, merged_df):
    """
    Calculates the r value (correlating business sales with foot traffic) for a
    merged dataframe based on the amount of regions
    Args:
        x - number of regions to split the city into (int)
        merged_df - Merged dataframe of business and foot traffic (DataFrame)
    Returns: correlation coefficient (float)
    """
    min_lat = min(list(merged_df["latitude"]))
    min_long = min(list(merged_df["longitude"]))
    max_lat = max(list(merged_df["latitude"]))
    max_long = max(list(merged_df["longitude"]))

    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # Find factors of x
    factors = [
        (i, x // i) for i in range(1, int(math.sqrt(x)) + 1) if x % i == 0
    ]
    # Choose the factors that are closest to each other
    factor1, factor2 = min(factors, key=lambda f: abs(f[0] - f[1]))
    # Calculate latitude and longitude step sizes for x regions
    lat_step = lat_range / factor1
    long_step = long_range / factor2

    # Define a function to assign region based on latitude and longitude
    def assign_region(row):
        """
        Function to assign region based on latitude and longitude.
        Args:
            row - row of a dataframe with latitude and longitude (DataFrame)
        Returns: Assigns row to a geographic region (float)
        """
        lat_region = int((row["latitude"] - min_lat) / lat_step) + 1
        long_region = int((row["longitude"] - min_long) / long_step) + 1
        return (lat_region - 1) * factor2 + long_region

    # Create a new column 'geographic_region' based on latitude and longitude
    merged_df["geographic_region"] = merged_df.apply(assign_region, axis=1)

    # Group by geographic region and aggregate data
    region_aggregate_df = (
        merged_df.groupby("geographic_region")
        .agg({"Sales Volume (9) - Location": "sum", "raw_visit_counts": "sum"})
        .reset_index()
    )

    # Calculate the linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        region_aggregate_df["Sales Volume (9) - Location"],
        region_aggregate_df["raw_visit_counts"],
    )
    return r_value


def r_plot(merged_df):
    """
    Outputs a line plot that shows differing correlation values depending on
    the amount of regions the city is split into
    Args:
        merged_df - Merged dataframe of business and foot traffic (DataFrame)
    Returns: line plot correlation coefficient vs number of regions (float)
    """
    # Initialize lists to store x values and r_values
    x_values = list(range(1, 140))
    r_values = []

    # Calculate r_value for each x
    for x in x_values:
        r_value = calculate_r_value(x, merged_df)
        r_values.append(r_value)

    # Plot the results
    plt.plot(x_values, r_values)
    plt.xlabel("Number of Regions (x)")
    plt.ylabel("R Value")
    plt.title("Effect of Region Count on R Value")
    plt.grid(True)
    plt.show(block=True)


def find_addresses(x, merged_df):
    """
    Find the address of the top business in every region based on the amount of
    regions the city is split into
    Args:
        x - number of regions to split the city into (int)
        merged_df - Merged dataframe of business and foot traffic (DataFrame)
    Returns: Printing all addresses
    """
    min_lat = min(list(merged_df["latitude"]))
    min_long = min(list(merged_df["longitude"]))
    max_lat = max(list(merged_df["latitude"]))
    max_long = max(list(merged_df["longitude"]))

    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # Find factors of x
    factors = [
        (i, x // i) for i in range(1, int(math.sqrt(x)) + 1) if x % i == 0
    ]

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
        """
        Function to assign region based on latitude and longitude.
        Args:
            row - row of a dataframe with latitude and longitude (DataFrame)
        Returns: Assigns row to a geographic region (float)
        """
        lat_region = int((row["latitude"] - min_lat) / lat_step) + 1
        long_region = int((row["longitude"] - min_long) / long_step) + 1
        return (lat_region - 1) * factor2 + long_region

    # Create a new column 'geographic_region' based on latitude and longitude
    merged_df["geographic_region"] = merged_df.apply(assign_region, axis=1)

    # Group by geographic region and find top sales in each region
    top_sales_in_regions = merged_df.groupby("geographic_region").apply(
        lambda group: group.nlargest(1, "Sales Volume (9) - Location")
    )

    # Print the street addresses of the businesses with
    # top sales in each region
    for index, row in top_sales_in_regions.iterrows():
        print(f"Region {index}: {row['street_address']}")


def find_top_businesses(business_df, foot_df, year, radius_km=1.0):
    """
    Find the top businesses in the city based on foot traffic
    surrounding it
    Args:
        business_df - business dataframe (DataFrame)
        foot_df - foot traffic dataframe (DataFrame)
        year - the year (int)
        radius_km - the radius around each business where foot traffic is
        calculated (float)
    Returns: Dataframe of top businesses (DataFrame)
    """
    # Filter foot traffic data for the specified year
    foot_df_year = filter_year(foot_df, year)

    # Group foot traffic data by business location and sum
    # up foot traffic counts
    foot_traffic_by_business = (
        foot_df_year.groupby(
            ["location_name", "latitude", "longitude", "street_address"]
        )["raw_visit_counts"]
        .sum()
        .reset_index()
    )

    top_businesses = []

    for _, business_row in business_df.iterrows():
        # Extract latitude and longitude of the current business
        business_lat = business_row["Latitude"]
        business_long = business_row["Longitude"]

        # Calculate the distance between the business and foot
        # traffic locations
        distances = cdist(
            [(business_lat, business_long)],
            foot_traffic_by_business[["latitude", "longitude"]],
            metric="euclidean",
        ).flatten()

        # Filter foot traffic locations within the specified radius
        foot_traffic_within_radius = foot_traffic_by_business[
            distances <= radius_km
        ]

        # Calculate total foot traffic in the area surrounding the business
        total_foot_traffic = foot_traffic_within_radius[
            "raw_visit_counts"
        ].sum()

        # Append the total foot traffic along with business
        # information to the list of top businesses
        top_businesses.append(
            {
                "Company": business_row["Company"],
                "Address Line 1": business_row["Address Line 1"],
                "Latitude": business_lat,
                "Longitude": business_long,
                "Total Foot Traffic": total_foot_traffic,
            }
        )

    # Convert the list of top businesses to a DataFrame
    top_businesses_df = pd.DataFrame(top_businesses)

    # Sort the DataFrame by total foot traffic in descending order
    top_businesses_df = top_businesses_df.sort_values(
        by="Total Foot Traffic", ascending=False
    )

    return top_businesses_df


def find_top_unique_business_addresses(
    business_df,
    foot_df,
    year,
    min_unique_businesses=100,
    max_radius_km=1.0,
    radius_step=0.1,
):
    """
    Find the top x unique businesses in the city based on foot traffic
    surrounding it with x being min_unique_businesses.
    Args:
        business_df - business dataframe (DataFrame)
        foot_df - foot traffic dataframe (DataFrame)
        year - the year (int)
        min_unique_businesses - minimum unique business locations
        to output (int)
        max_radius_km - the max radius around each business where foot traffic
        is calculated (float)
        radius_step - the step the radius will decrease by in each loop (float)
    Returns: Dataframe of top businesses (DataFrame)
    """

    unique_businesses = set()
    radius_km = max_radius_km  # Starting radius

    while radius_km > 0:
        top_businesses_df = find_top_businesses(business_df, foot_df, year)

        # Add unique businesses to the set
        unique_businesses.update(
            top_businesses_df.drop_duplicates(subset="Total Foot Traffic")[
                "Address Line 1"
            ]
        )

        # Check if enough unique businesses are found
        if len(unique_businesses) >= min_unique_businesses:
            break

        # Increase radius for the next iteration
        radius_km -= radius_step

    # Return the street addresses of the top unique businesses
    return top_businesses_df[
        top_businesses_df["Address Line 1"].isin(unique_businesses)
    ]["Address Line 1"].tolist()


def morph_and_visualize_business(df):
    """
    Restructures a given dataset and outputs a visualization of business
    locations in the city.
    Args:
        df - dataframe of business locations (DataFrame)
    Returns: Visualization of business locations (plot)
    """
    # Make a subset of the dataset
    columns_to_keep = ["Latitude", "Longitude", "Total Foot Traffic"]
    df_subset = df[columns_to_keep].copy()

    joint_axes = sns.jointplot(
        x="Longitude",
        y="Latitude",
        hue="Total Foot Traffic",
        data=df_subset,
        s=0.5,
    )

    cx.add_basemap(
        joint_axes.ax_joint,
        crs="EPSG:4326",
        source=cx.providers.CartoDB.PositronNoLabels,
    )

    # Adjust point sizes in the legend
    handles, labels = joint_axes.ax_joint.get_legend_handles_labels()

    # Extract colors from existing legend handles
    legend_colors = [handle.get_color() for handle in handles]

    # Create custom legend handles with both color and size
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markersize=10,
            markerfacecolor=color,
        )
        for color in legend_colors
    ]

    # Add legend with custom handles
    joint_axes.ax_joint.legend(
        legend_handles, labels, bbox_to_anchor=(1.5, 1.1), title="Visit Counts"
    )

    # Increases size of points in the graph
    joint_axes.ax_joint.collections[0].set_sizes([20])

    plt.show(block=True)
