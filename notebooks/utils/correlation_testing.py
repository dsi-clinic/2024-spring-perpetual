"""Functions to supplement Safegraph correlation tests.
"""

# Standard library imports
import math
import warnings

# Third-party imports
import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from scipy.stats import linregress

# Application imports
from .safegraph import filter_by_year

warnings.filterwarnings("ignore")


def merge(
    business_df: pd.DataFrame, foot_df: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Merges Infogroup business data and Safegraph foot traffic
    data for a particular year.

    Args:
        business_df (`pd.DataFrame`): The Infogroup dataset.

        foot_df (`pd.DataFrame`): The Safegraph dataset.

        year (`int`): The year by which to filter.

    Returns:
        (`pd.DataFrame`): The merged DataFrame.
    """
    foot_df_year = filter_by_year(foot_df, year)
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


def calculate_r_value(x: int, merged_df: pd.DataFrame) -> float:
    """Calculates the r value (correlating business sales with foot traffic)
    for a merged DataFrame based on the numbers of regions.

    Args:
        x (`int`): The number of regions to split the city into.

        merged_df (`pd.DataFrame`): The merged DataFrame of
            business and foot traffic.

    Returns:
        (`float`): The correlation coefficient.
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


def r_plot(merged_df: pd.DataFrame) -> None:
    """Outputs a line plot that shows differing correlation values depending on
    the amount of regions the city is split into.

    Args:
        merged_df (`pd.DataFrame`): A merged DataFrame of
            business and foot traffic.

    Returns:
        `None`
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


def find_addresses(x: int, merged_df: pd.DataFrame) -> None:
    """Find the address of the top business in every region
    based on the number of regions the city is split into
    and then prints those addresses.

    Args:
        x (`int`): The number of regions to split the city into.

        merged_df (`pd.DataFrame`): The merged DataFrame of
            business and foot traffic.

    Returns:
        `None`
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


def find_top_businesses(
    business_df: pd.DataFrame,
    foot_df: pd.DataFrame,
    year: int,
    radius_km: float = 1.0,
) -> pd.DataFrame:
    """Finds the top businesses in the city based on foot traffic surrounding it.

    Args:
        business_df (`pd.DataFrame`): The business DataFrame.

        foot_df (`pd.DataFrame`): The foot traffic DataFrame.

        year (`int`): The year.

        radius_km (`float`): The radius around each business where
            foot traffic is calculated.

    Returns:
        (`pd.DataFrame): The top businesses.
    """
    # Filter foot traffic data for the specified year
    foot_df_year = filter_by_year(foot_df, year)

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
    business_df: pd.DataFrame,
    foot_df: pd.DataFrame,
    year: int,
    min_unique_businesses: int = 100,
    max_radius_km: float = 1.0,
    radius_step: float = 0.1,
) -> pd.DataFrame:
    """Finds the top x unique businesses in the city based on foot traffic
    surrounding it with x being min_unique_businesses.

    Args:
        business_df (`pd.DataFrame`): The business DataFrame.

        foot_df (`pd.DataFrame`): The foot traffic DataFrame.

        year (`int`): The year.

        min_unique_businesses (`int`): The minimum unique business
            locations to output.

        max_radius_km (`float`): The max radius around each business
            where foot traffic is calculated.

        radius_step (`float`): The step the radius will decrease by in each loop.

    Returns:
        (`pd.DataFrame`): The top businesses.
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


def morph_and_visualize_business(df: pd.DataFrame) -> None:
    """Restructures a given dataset and outputs a visualization
    of business locations in the city.

    Args:
        df (`pd.DataFrame`): The business locations.

    Returns:
        `None`
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
