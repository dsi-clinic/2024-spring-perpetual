"""Functions used to process and visualize Safegraph foot traffic data.
"""

# Standard library imports
import itertools
import json
import os
import webbrowser
from typing import List, Optional

# Third-party imports
import contextily as cx
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polyline
import requests
import seaborn as sns
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.lines import Line2D

# Application imports
from .constants import DATA_DIR

SAFEGRAPH_RELEVANT_COLUMNS = [
    "sub_category",
    "safegraph_place_id",
    "location_name",
    "latitude",
    "longitude",
    "raw_visitor_counts",
    "date_range_start",
    "date_range_end",
    "raw_visit_counts",
    "raw_visitor_counts",
    "related_same_day_brand",
]


def preview_dataset(df: pd.DataFrame, num_rows: int = 5) -> None:
    """Prints the DataFrame's total row and column counts, column
    names, and a preview of its starting rows.

    Args:
        df (`pd.DataFrame`): The dataset.

        num_rows (`int`): The number of starting rows to
            include in the preview. Defaults to 5.

    Returns:
        `None`
    """
    # Display count
    display(HTML("<b>Count</b>"))
    display(f"{len(df):,} record(s) and {len(df.columns):,} column(s)")

    # List columns
    display(HTML("<b>Columns</b>"))
    display(df.columns)

    # Preview relevant columns
    display(HTML("<b>Preview Relevant Columns</b>"))
    display(df[SAFEGRAPH_RELEVANT_COLUMNS].head(num_rows))


def summarize_column_ranges(df: pd.DataFrame) -> None:
    """Prints the range and percentiles of the values in
    each of the DataFrame's relevant columns.

    Args:
        df (`pd.DataFrame`): The dataset.

    Returns:
        `None`
    """
    # Numerical columns
    display(HTML(f"<b>Basic Stats for Numerical Columns</b>"))
    display(df[SAFEGRAPH_RELEVANT_COLUMNS].describe())

    # Datetimes
    for col in ("date_range_start", "date_range_end"):
        display(HTML(f"<b>Range for {col}</b>"))
        display(df.query(f"{col} == {col}")[col].agg(["min", "max"]))


def explode_dataset(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Explodes the SafeGraph data so every row represents
    a single visit at a location, rather than a total of
    visits in a particular month at a location, and then
    casts the dataset to a GeoDataFrame with a CRS of
    EPSG:4326.

    Args:
        df (`pd.DataFrame`): The input DataFrame.

    Returns:
        (`gpd.GeoDataFrame`): The transformed data.
    """
    # Subset DataFrame
    subset_df = df[SAFEGRAPH_RELEVANT_COLUMNS].copy()

    # Fill NaNs in the raw_visit_counts column with zero
    # and make all entries in that column integers
    subset_df["raw_visit_counts"] = (
        pd.to_numeric(subset_df["raw_visit_counts"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Use repeat to expand the DataFrame
    subset_df = subset_df.reset_index(drop=True)
    indices = subset_df.index.repeat(subset_df["raw_visit_counts"])
    expanded_df = subset_df.loc[indices].reset_index(drop=True)

    # Assert that the number of rows is equal to the sum of raw visit counts
    assert len(expanded_df) == subset_df["raw_visit_counts"].sum()

    # Check expanded_df for missing or infinite values and
    # drop them if necessary
    expanded_df = expanded_df.dropna(subset=["longitude", "latitude"])
    expanded_df = expanded_df[
        (expanded_df["longitude"] != np.inf)
        & (expanded_df["latitude"] != np.inf)
    ]
    expanded_df = expanded_df[
        (expanded_df["longitude"] != -np.inf)
        & (expanded_df["latitude"] != -np.inf)
    ]

    # Convert to GeoDataFrame and set CRS
    gdf = gpd.GeoDataFrame(
        expanded_df,
        geometry=gpd.points_from_xy(
            x=expanded_df["longitude"], y=expanded_df["latitude"]
        ),
        crs="EPSG:4326",
    )

    return gdf[
        [
            "date_range_start",
            "date_range_end",
            "latitude",
            "longitude",
            "geometry",
        ]
    ]


def get_top_location_categories(df: pd.DataFrame, n: int = 20) -> None:
    """Fetches the top `N` locations in terms of visit counts.

    Args:
        df (`pd.DataFrame`): The dataset.

        n (`int`): The top `N` locations to pull.

    Returns:
        `None`
    """
    ranked_locations = (
        df[["sub_category", "raw_visit_counts"]]
        .groupby(by=["sub_category"])
        .sum()
        .reset_index()
        .sort_values(by="raw_visit_counts", ascending=False)
        .reset_index(drop=True)
        .head(n)
    )
    ranked_locations["raw_visit_counts"] = ranked_locations[
        "raw_visit_counts"
    ].apply(lambda c: f"{int(c):,}")
    display(ranked_locations)


def get_top_locations_with_related_brands(
    data: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """Fetches the top `N` locations in terms of visit counts,
    along with their top related same-day brand.

    Args:
        df (`pd.DataFrame`): The dataset.

        n (`int`): The top `N` locations to pull.

    Returns:
        `None`
    """
    # Step 1: Identify the top N businesses by visit counts
    # Now using 'safegraph_place_id' to ensure unique identification of places
    top_visited = data.sort_values(
        by="raw_visit_counts", ascending=False
    ).drop_duplicates(subset="safegraph_place_id")
    top_visited = top_visited.head(n)

    # Initialize a list to hold the result
    results = []

    # Step 2: For each of the top 10 businesses,
    # find the top related brand based on visit counts
    for _, business in top_visited.iterrows():

        # Parse the JSON data in the 'related_same_day_brand' column
        try:
            related_brands = json.loads(business["related_same_day_brand"])
        except json.JSONDecodeError:
            related_brands = {}

        related_brands_df = pd.DataFrame(
            list(related_brands.items()), columns=["Brand", "Count"]
        )
        top_related = related_brands_df.sort_values(
            by="Count", ascending=False
        ).head(1)

        for _, row in top_related.iterrows():
            # Find the matching business entry
            # for the related brand using 'location_name'
            # This assumes that 'location_name' can still uniquely
            # identify related brands for fetching latitude and longitude
            related_brand_info = data[
                data["location_name"] == row["Brand"]
            ].iloc[0]
            result = {
                "Safegraph Place ID": business["safegraph_place_id"],
                "Main Business": business["location_name"],
                "Main Latitude": business["latitude"],
                "Main Longitude": business["longitude"],
                "Related Brand": row["Brand"],
                "Related Brand Latitude": related_brand_info["latitude"],
                "Related Brand Longitude": related_brand_info["longitude"],
            }
            results.append(result)

    return pd.DataFrame(results)


def compute_fastest_routes(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Fix algorithm for computing fastest route and update documentation."""
    routes = []
    osrm_url = "http://router.project-osrm.org/route/v1/foot/"

    for _, row in df.iterrows():
        if (
            pd.notna(row["Main Latitude"])
            and pd.notna(row["Main Longitude"])
            and pd.notna(row["Related Brand Latitude"])
            and pd.notna(row["Related Brand Longitude"])
        ):
            request_url = f"{osrm_url}{row['Main Longitude']}, {row['Main Latitude']};{row['Related Brand Longitude']},{row['Related Brand Latitude']}?overview=full"
            # 'full' for complete route geometry
            try:
                response = requests.get(request_url)
                response.raise_for_status()
                route_data = response.json()

                if "routes" in route_data and route_data["routes"]:
                    first_route = route_data["routes"][0]
                    geometry = first_route.get(
                        "geometry"
                    )  # This is often an encoded polyline
                    decoded_geometry = polyline.decode(
                        geometry
                    )  # Decoding the polyline to a list of (lat, lon) tuples

                    route_info = {
                        "Main Business": row["Main Business"],
                        "Related Brand": row["Related Brand"],
                        "Distance": first_route["distance"],
                        "Duration": first_route["duration"],
                        "Geometry": decoded_geometry, # Store the decoded geometry
                    }
                    routes.append(route_info)
                else:
                    print(
                        "No route found for" + 
                        f"{row['Main Business']} to {row['Related Brand']}"
                    )
            except requests.RequestException as e:
                print(f"Request failed: {e}")

    return pd.DataFrame(routes)


def plot_routes(
    df_routes: pd.DataFrame,
    output_file_name: Optional[str] = None,
) -> None:
    """Plots routes on a map, with an option to save
    the map and open it in a web browser.

    Args:
        df_routes (`pd.DataFrame`): A DataFrame containing routes information.

        output_file_name (`str`): The name of the output file, saved
            under "data/foot-traffic/output". Defaults to `None`, in
            which case the map is rendered but not saved.

    Returns:
        `None`
    """
    # Define a list of colors for different routes
    colors = itertools.cycle(
        [
            "blue",
            "green",
            "red",
            "purple",
            "orange",
            "darkblue",
            "lightgreen",
            "gray",
            "black",
            "pink",
        ]
    )

    # Create the base map
    if df_routes.empty:
        print("No routes to display.")
        return None

    first_coord = df_routes.iloc[0]["Geometry"][0]
    fmap = folium.Map(location=list(first_coord), zoom_start=12)

    # Add a legend to the map
    legend_html = """
    <div style="position: fixed;
                bottom: 50px; left: 50px; width: 150px; height: 90px;
                border:2px solid grey; z-index:9999; font-size:14px;
                ">&nbsp; <b>Legend</b> <br>
                    &nbsp; Main Business &nbsp;
                    <i class="fa fa-map-marker fa-2x" style="color:red"></i><br>
                    &nbsp; Related Brand &nbsp;
                    <i class="fa fa-map-marker fa-2x" style="color:blue"></i>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Add the routes to the map
    for _, row in df_routes.iterrows():
        route_color = next(colors)  # Get a color from the cycle
        popup_text = (
            f"Route from {row['Main Business']} to {row['Related Brand']}"
        )
        folium.PolyLine(
            locations=[(coord for coord in row["Geometry"])],
            weight=5,
            color=route_color,
            popup=folium.Popup(
                popup_text, parse_html=True
            ),  # Add popup to the route
        ).add_to(fmap)

        # Add markers for the start (main business) and
        # end (related brand) points
        folium.Marker(
            location=[*row["Geometry"][0]],
            popup=f"{row['Main Business']}",
            icon=folium.Icon(color="red"),
        ).add_to(fmap)

        folium.Marker(
            location=[*row["Geometry"][-1]],
            popup=f"{row['Related Brand']}",
            icon=folium.Icon(color="blue"),
        ).add_to(fmap)

    # Save the map if specified
    if output_file_name:
        output_dir = f"{DATA_DIR}/foot-traffic/output"
        os.makedirs(output_dir, exist_ok=True)
        output_file_name = (
            output_file_name
            if output_file_name.endswith(".html")
            else output_file_name + ".html"
        )
        output_file_path = os.path.join(output_dir, output_file_name)
        fmap.save(output_file_path)
        webbrowser.open(f"file://{os.path.abspath(output_file_path)}")
    else:
        temp_file_path = "/tmp/temp_map.html"
        fmap.save(temp_file_path)
        webbrowser.open(f"file://{os.path.abspath(temp_file_path)}")

    display(fmap)


def plot_top_restaurants(df: pd.DataFrame) -> None:
    """Plots a choropleth map of restaurant locations,
    with the color reflecting the total number of raw visits.

    Args:
        df (`pd.DataFrame`): The foot traffic data.

    Returns:
        `None`
    """
    # Filter data to include only restaurant establishments and sort values
    food_df = df[df["top_category"] == "Restaurants and Other Eating Places"]
    food_sorted_df = food_df.sort_values(by="raw_visit_counts", ascending=True)
    food_sorted_df.reset_index(drop=True, inplace=True)
    food_grpd_df = (
        food_sorted_df.groupby(
            by=["safegraph_place_id", "latitude", "longitude"]
        )
        .agg({"raw_visit_counts": "sum"})
        .reset_index()
    )

    # Initialize map
    joint_axes = sns.jointplot(
        x="longitude",
        y="latitude",
        hue="raw_visit_counts",
        xlim=(-155.1, -155.04),
        ylim=(19.68, 19.735),
        data=food_grpd_df,
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

    # Render plot
    display(plt.show(block=True))


def split_into_months(
    df: pd.DataFrame, city: str, persist: bool = False
) -> List[pd.DataFrame]:
    """Takes a Safegraph foot traffic DataFrame and splits it into
    12 DataFrames, one for each month of the year. Optionally
    saves each DataFrame to a file under "data/foot-traffic/output".

    Args:
        df (`pd.DataFrame`): The input DataFrame containing 'date_range_start'.

        city (`str`): The name of the city to use in the output file names.

        persist (`bool`): A boolean indicating whether the DataFrame
            should be saved to local file storage.

    Returns:
        (`list` of `pd.DataFrame`): The month-based DataFrames.
    """
    # Extract month from the date
    df["month"] = df["date_range_start"].str[5:7].astype("Int64")

    # Create a directory for saving the output
    output_dir = f"{DATA_DIR}/foot-traffic/output"
    os.makedirs(output_dir, exist_ok=True)

    # Create and save a dataframe for each month
    df_lst = []
    for month in range(1, 13):
        month_df = df[df["month"] == month]
        df_lst.append(month_df)
        if persist:
            output_file_name = f"{city}_month_{month}.csv"
            output_file_path = os.path.join(output_dir, output_file_name)
            month_df.to_csv(output_file_path, index=False)
    return df_lst


def x_highest_visits(df, x):
    """
    Identifies locations with the highest raw visit counts.
    Args:
        df - dataframe of foot traffic data (DataFrame)
        x - specified number of locations to be returned (int)
    Returns: list of x number of locations (list of strings)
    """
    return df.sort_values(by="raw_visit_counts", ascending=False)[
        "location_name"
    ].unique()[:x]


def split_into_seasons(df):
    """
    Splits a dataframe into seasons
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: One dataframe for each season (4 dataframes)
    """
    df["month"] = df["date_range_start"].str[5:7].astype("Int64")
    df_winter = df[df["month"] < 4]
    df_spring = df[(df["month"] > 3) & (df["month"] < 7)]
    df_summer = df[(df["month"] > 6) & (df["month"] < 10)]
    df_fall = df[df["month"] > 9]
    return df_winter, df_spring, df_summer, df_fall


def morph_and_visualize(df):
    """
    Restructures a given dataset and outputs a visualization of city foot
    traffic.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: Visualization of foot traffic data (plot)
    """
    # Restructure the dataset so each row represents a place visit
    # Make a subset of the dataset
    columns_to_keep = [
        "placekey",
        "location_name",
        "latitude",
        "longitude",
        "raw_visit_counts",
        "raw_visitor_counts",
        "related_same_day_brand",
    ]
    df_subset = df[columns_to_keep].copy()

    # Fill NaNs in the raw_visit_counts column with 0 and
    # make all entries in that column integers
    df_subset["raw_visit_counts"] = (
        pd.to_numeric(df_subset["raw_visit_counts"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df_sorted = df.sort_values(by="raw_visit_counts", ascending=True)
    df_sorted.reset_index(drop=True, inplace=True)

    joint_axes = sns.jointplot(
        x="longitude",
        y="latitude",
        hue="raw_visit_counts",
        data=df_sorted,
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


def food_df(df):
    """
    Filters dataset for only food locations.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: A dataframe with only food locations (1 dataframe)
    """
    return df[df["top_category"] == "Restaurants and Other Eating Places"]


def create_location_df(dataframe, location_name):
    """
    Filters dataset for a specific location name.
    Args:
        df - dataframe of foot traffic data (DataFrame)
        location_name - name of a location (string)
    Returns: A dataframe with only locations that match the name given
    (1 dataframe)
    """
    # Filter for the given location_name
    location_filter = dataframe[dataframe["location_name"] == location_name]

    # Select only the relevant columns
    location_df = location_filter[
        ["location_name", "latitude", "longitude"]
    ].copy()

    # Drop rows with NaN values in 'latitude' or 'longitude'
    location_df = location_df.dropna(subset=["latitude", "longitude"])

    # Drop duplicate rows based on 'latitude' and 'longitude'
    # to ensure unique locations
    unique_location_df = location_df.drop_duplicates(
        subset=["latitude", "longitude"]
    )

    return unique_location_df


def top_x_businesses_and_related(df, x):
    """
    Make visualization with the top x businessed and their corresponding top
    related brands.
    Args:
        df - dataframe of foot traffic data (DataFrame)
    Returns: visualization with the top x businessed and
    their corresponding top related brands
    """
    # Create a df with top x most visited businesses
    # and their corresponding locations

    # Create an empty df
    top_visits = pd.DataFrame()

    # Get the names of the top x most visited businesses
    top_business_names = df.sort_values(by="raw_visit_counts",
                                        ascending=False)[
        "location_name"
    ].unique()[:x]

    # Temporary list to collect DataFrames
    temp_dfs = []

    for business in top_business_names:
        business_data = df[df["location_name"] == business]
        # Sort and take the first row with the highest "raw_visit_counts"
        top_business_data = business_data.sort_values(
            by="raw_visit_counts", ascending=False
        ).head(1)
        # Collect the top row for each business
        temp_dfs.append(top_business_data)

    # Concatenate all the top rows into a single DataFrame
    top_visits = pd.concat(temp_dfs)

    # Reset index after concatenating
    top_visits.reset_index(drop=True, inplace=True)

    # Drop any rows with NaN values in 'latitude' or 'longitude'
    top_visits = top_visits.dropna(subset=["latitude", "longitude"])

    top_related_brands = {}

    for business in top_business_names:
        # Filter for the current business
        business_data = df[df["location_name"] == business].copy()

        # Safely convert the JSON string in related_same_day_brand to a list,
        # handling None values
        business_data["related_same_day_brand_list"] = business_data[
            "related_same_day_brand"
        ].apply(lambda x: json.loads(x) if x is not None else [])

        # Explode the DataFrame so each brand has its own row
        all_related_brands = business_data.explode(
            "related_same_day_brand_list"
        )

        # Count the most common related same-day brand
        if not all_related_brands.empty:
            top_brand = (
                all_related_brands["related_same_day_brand_list"]
                .value_counts()
                .nlargest(3)
            )
        else:
            top_brand = "No data"

        # Store the result in the dictionary
        top_related_brands[business] = top_brand

    # List of brand DFs
    df_list = []

    for _, brand in top_related_brands.items():
        for b in brand.keys():
            df_list.append(create_location_df(df, b))

    top_related_df = pd.concat(df_list, ignore_index=True)

    # Convert df to Geo df
    gdf = gpd.GeoDataFrame(
        top_visits,
        geometry=gpd.points_from_xy(top_visits.longitude, top_visits.latitude),
    )
    gdf_related_brands = gpd.GeoDataFrame(
        top_related_df,
        geometry=gpd.points_from_xy(
            top_related_df.longitude, top_related_df.latitude
        ),
    )

    # Set CRS to WGS84
    gdf.crs = "EPSG:4326"
    gdf_related_brands.crs = "EPSG:4326"

    # Convert to Web Mercator for mapping with contextily basemap
    gdf = gdf.to_crs(epsg=3857)
    gdf_related_brands = gdf_related_brands.to_crs(epsg=3857)

    # Basic plot of related same day brand visits and
    # top x highest foot traffic businesses (w/ labels)

    # Set the figure size
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot top locations
    gdf.plot(ax=ax, color="blue", marker="o", label="Top Locations")

    # Plot related brands
    gdf_related_brands.plot(
        ax=ax, color="red", marker="x", label="Related Brands"
    )

    # Add basemap
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)

    # Add labels for top locations
    for idx, row in gdf.iterrows():
        name = row["location_name"]
        # Check if the name has already been labeled
        ax.text(
            row.geometry.x,
            row.geometry.y,
            s=name,
            fontsize=8,
            ha="right",
            va="bottom",
            color="blue",
        )

    # Add labels for related brands, ensuring each name is shown only once
    for idx, row in gdf_related_brands.iterrows():
        name = row["location_name"]
        # Only label if the name hasn't been labeled yet
        ax.text(
            row.geometry.x,
            row.geometry.y,
            s=name,
            fontsize=8,
            ha="left",
            va="top",
            color="red",
        )

    ax.set_axis_off()
    plt.legend()
    plt.show(block=True)
