"""Functions used to process and visualize Infogroup business data.
"""

# Standard library imports
import io

# Third-party imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Application imports
from .constants import INFOGROUP_2023_FPATH


def get_infogroup_city_records(
    fpath: str, city: str, state: str
) -> pd.DataFrame:
    """Opens the file and filters the data to include
    only records that match the given city and state.
    NOTE: There are instances of cities with the same
    name in the same city, in which case additional steps
    are needed to find the exact city of interest.

    Args:
        fpath (`str`): The path to the Infogroup data file.

        city (`str`): The name of the city to fetch.

        state (`str`): The state that identifies the city.
            Passed as a two-letter abbreviation.

    Returns:
        (`pd.DataFrame`): The output DataFrame containing
            businesses in the city, along with metadata.
    """
    # Initialize variables
    whole_df = None
    lines_left = True

    # Open file
    with open(fpath, encoding="ISO-8859-1") as f:
        # Read header (first line)
        header = f.readline()

        # Process remaining lines in batches
        while True:
            # Get batch of lines with header
            lines = [header]
            for _ in range(10_000):
                line = f.readline()
                if not line:
                    lines_left = False
                    break
                lines.append(line)

            # Convert to in-memory CSV file and then read into DataFrame
            df = pd.read_csv(io.StringIO("\n".join(lines)))

            # Filter to desired city and state
            df_filtered = df[
                (df["CITY"] == city.upper()) & (df["STATE"] == state.upper())
            ]

            # Store results in final DataFrame if any city records found
            if len(df_filtered) > 0:
                whole_df = (
                    df_filtered
                    if whole_df is None
                    else pd.concat([whole_df, df_filtered])
                )

            # If end of file reached, return final DataFrame
            if not lines_left:
                return whole_df


def format_infogroup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Subsets an Infogroup dataset to relevant columns,
    renames those columns, drops duplicates, and resets
    the index.

    Args:
        df (`pd.DataFrame`): The input Infogroup dataset.

    Returns:
        (`pd.DataFrame`): The formatted dataset.
    """
    df = df.loc[
        :,
        [
            "COMPANY",
            "CITY",
            "ADDRESS LINE 1",
            "LATITUDE",
            "LONGITUDE",
            "SALES VOLUME (9) - LOCATION",
            "EMPLOYEE SIZE (5) - LOCATION",
            "PARENT ACTUAL SALES VOLUME",
        ],
    ]
    df = df.rename(
        columns={
            "ADDRESS LINE 1": "street1",
            "CITY": "city",
            "COMPANY": "name",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "SALES VOLUME (9) - LOCATION": "sales_volume",
            "EMPLOYEE SIZE (5) - LOCATION": "employee_size",
            "PARENT ACTUAL SALES VOLUME": "parent_sales_volume",
        }
    )
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)

    return df


def load_infogroup_data(city: str, state: str) -> pd.DataFrame:
    """Loads and cleans Infogroup data for a city.

    Args:
        city (`str`): The name of the city to fetch.

        state (`str`): The state that identifies the city.
            Passed as a two-letter abbreviation.

    Returns:
        (`pd.DataFrame`): The Infogroup data for the city.
    """
    city = city.upper().strip()
    state = state.upper().strip()
    info_df = get_infogroup_city_records(INFOGROUP_2023_FPATH, city, state)
    business_df = format_infogroup_df(info_df)
    return business_df


def aggregate_stats_by_region(
    info_gdf: gpd.GeoDataFrame, safegraph_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Aggregates statistics by region.

    Args:
        info_gdf (`gpd.GeoDataFrame`): The business sales data.

        safegraph_gdf (`gpd.GeoDataFrame`): The foot traffic data.

    Returns:
        (`pd.DataFrame`): The aggregated dataset.
    """
    # Find minimum latitude and longitude across datasets
    info_minx, info_miny, info_maxx, info_maxy = info_gdf.total_bounds
    sgph_minx, sgph_miny, sgph_maxx, sgph_maxy = safegraph_gdf.total_bounds
    min_lon = min(info_minx, sgph_minx)
    min_lat = min(info_miny, sgph_miny)
    max_lon = max(info_maxx, sgph_maxx)
    max_lat = max(info_maxy, sgph_maxy)

    # Calculate latitude and longitude ranges
    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    # Calculate latitude and longitude step sizes for 20 regions
    lat_step = lat_range / 5  # Since we want 5 regions along latitude
    lon_step = lon_range / 4  # Since we want 4 regions along longitude

    # Define local function to assign region based on latitude and longitude
    def assign_region(row: pd.Series) -> int:
        """Assigns a region number based on the Series' coordinates.

        Args:
            row (`pd.Series`): The row.

        Returns:
            (`int`): The region number.
        """
        lat_region = int((row["latitude"] - min_lat) / lat_step) + 1
        lon_region = int((row["longitude"] - min_lon) / lon_step) + 1
        return (lat_region - 1) * 4 + lon_region

    # Aggregate sales volume by region
    col_map = {
        "COMPANY": "name",
        "SALES VOLUME (9) - LOCATION": "sales_volume",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "geometry": "geometry",
    }
    info_regions_df = info_gdf.copy()[col_map.keys()].rename(columns=col_map)
    info_regions_df["geographic_region"] = info_regions_df.apply(
        assign_region, axis=1
    )
    region_sales_df = (
        info_regions_df.groupby("geographic_region")
        .agg({"sales_volume": "sum"})
        .reset_index()
        .loc[:, ["geographic_region", "sales_volume"]]
    )

    # Aggregate foot traffic by region
    cols = [
        "location_name",
        "raw_visit_counts",
        "latitude",
        "longitude",
        "geometry",
    ]
    safegraph_regions_df = safegraph_gdf[cols].rename(
        columns={"location_name": "name"}
    )
    safegraph_regions_df = safegraph_regions_df.query(
        "(latitude == latitude) & (longitude == longitude)"
    )
    safegraph_regions_df["geographic_region"] = safegraph_regions_df.apply(
        assign_region, axis=1
    )
    foot_traffic_df = (
        safegraph_regions_df.groupby("geographic_region")
        .agg({"raw_visit_counts": "sum"})
        .reset_index()
        .loc[:, ["geographic_region", "raw_visit_counts"]]
    )

    # Merge DataFrames
    merged_df = region_sales_df.merge(
        foot_traffic_df, how="outer", on="geographic_region"
    )

    # Apply logarithmic transformation to the data
    merged_df["log_raw_visit_counts"] = np.log1p(merged_df["raw_visit_counts"])
    merged_df["log_sales_volume"] = np.log1p(merged_df["sales_volume"])

    # Reshape DataFrame for display
    merged_df = (
        merged_df.sort_values(by="geographic_region")
        .reset_index(drop=True)
        .replace({np.nan: 0})
    )
    merged_df["sales_volume"] = merged_df["sales_volume"].astype(int)
    merged_df["raw_visit_counts"] = merged_df["raw_visit_counts"].astype(int)

    return merged_df


def plot_region_stat_correlation(df: pd.DataFrame):
    """Plots a scatterplot of sales volume against visit
    counts with a line of best fit.

    Args:
        df (`pd.DataFrame): The data to plot.

    Returns:
        `None`
    """
    # Calculate the linear regression line
    slope, intercept, r_value, _, _ = linregress(
        df["log_sales_volume"], df["log_raw_visit_counts"]
    )
    x_values = np.linspace(
        min(df["log_sales_volume"]), max(df["log_sales_volume"]), 100
    )
    y_values = slope * x_values + intercept

    # Create a scatter plot with logarithmic transformation
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["log_sales_volume"],
        df["log_raw_visit_counts"],
        color="blue",
        alpha=0.5,
    )

    # Add the trendline
    plt.plot(
        x_values, y_values, color="red", label=f"Trendline (r={r_value:.2f})"
    )

    # Add labels and title
    plt.title(
        "Correlation between Log Sales Volume and Log Visit Counts by"
        " Geographic Region"
    )
    plt.xlabel("Log Sales Volume")
    plt.ylabel("Log Raw Visit Counts")

    # Show the plot
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
