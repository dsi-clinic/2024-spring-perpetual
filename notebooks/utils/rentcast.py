"""Provides functions used when processing Rentcast API data.
"""

# Standard library imports
import json
from typing import Tuple

# Third-party imports
import branca
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from IPython.display import display, HTML
from scourgify import normalize_address_record


RENTCAST_RELEVANT_COLUMNS = [
    "id",
    "formattedAddress",
    "latitude",
    "longitude",
    "yearBuilt",
    "propertyType",
    "squareFootage",
    "lotSize",
    "ownerOccupied",
    "lastSalePrice",
    "bathrooms",
    "bedrooms",
    "features",
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

    # Display null counts
    subset = df[RENTCAST_RELEVANT_COLUMNS]
    display(HTML("<b>Null Counts of Relevant Columns</b>"))
    display(subset.info())

    # Describe distribution of numerical columns
    display(HTML("<b>Column Value Distributions</b>"))
    display(subset.describe())

    # Preview relevant columns
    display(HTML("<b>Preview Relevant Columns</b>"))
    display(subset.head(num_rows))


def draw_histogram(
    series: pd.Series, title: str, x_label: str, bins: int = 100
) -> None:
    """Creates a histogram of the series' values.

    Args:
        series (`pd.Series`): The data to plot.

        title (`str`): The title of the plot.

        x_label (`str`): The label for the x-axis.

        bins (`int`): The number of bins to use. Defaults to 100.

    Returns:
        `None`
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(series, kde=True, color="blue", bins=30)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    display(plt.show())


def draw_scatterplot(
    df: pd.DataFrame,
    title: str,
) -> None:
    """Draws a scatterplot of the data.

    Args:
        df (`pd.DataFrame`): The data source holding
            all columns to plot.

        title (`str`): The title of the plot.

    Returns:
        `None`
    """
    plt.figure(figsize=(10, 6))
    g = sns.pairplot(df, height=2.5)
    g.map(sns.regplot)
    display(HTML(f"<b>{title}</b>"))
    plt.show()


def parse_street_address(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
    """Normalizes an address record and then parses its fields into five subcomponents.

    Args:
        df (`pd.DataFrame`): The data source.

        address_col (`str`): The name of the column holding raw addresses.

    Returns:
        (`pd.DataFrame`): A copy of the original DataFrame with
            five additional columns parsed from the original address:
            (1) street address line one, (2) street address line two,
            (3) city, (4) state, and (5) postal code.
    """

    # Define local function to normalize and parse street addresses
    def _parse_street_address(location: str) -> Tuple[str, str, str, str, str]:
        """Normalizes an address record and then parses
        its fields into five subcomponents.

        Args:
            location (`str`): The location to normalize and parse.

        Returns:
            (`str`, `str`, `str`, `str`, `str`): A five-item tuple consisting
                of street address line one, street address line two, city,
                state, and postal code.
        """
        try:
            return normalize_address_record(location).values()
        except:
            return (None, None, None, None, None)

    # Create new DataFrame with additional street address components
    copy = df.copy()
    new_cols = ["address_line_1", "address_line_2", "city", "state", "postal_code"]
    copy[new_cols] = copy[[address_col]].apply(
        lambda r: _parse_street_address(r[address_col]), axis=1, result_type="expand"
    )

    return copy


def view_against_yelp(rentcast_df: pd.DataFrame, yelp_df: pd.DataFrame) -> None:
    """Plots Rentcast buildings against points of interest fetched from Yelp
    on an interactive map to highlight their overlaps.

    Args:
        rentcast_df (`pd.DataFrame`): The Rentcast buildings.

        yelp_df (`pd.DataFrame`): The Yelp locations.

    Returns:
        `None`
    """
    # Transform Yelp DataFrame
    cols = ["Latitude", "Longitude"]
    yelp_df = yelp_df.reset_index().rename(columns={"index": "yelp_idx"})
    yelp_df[cols] = yelp_df[cols].astype(float)
    yelp_gdf = gpd.GeoDataFrame(
        yelp_df,
        geometry=gpd.points_from_xy(x=yelp_df["Longitude"], y=yelp_df["Latitude"]),
        crs="EPSG:4326",
    )
    yelp_gdf = yelp_gdf[
        ["yelp_idx", "Subject", "Location", "address_line_1", "geometry"]
    ]

    # Transform Rentcast DataFrame
    rentcast_df = rentcast_df.reset_index().rename(columns={"index": "rentcast_idx"})
    rentcast_df = rentcast_df.drop_duplicates(subset="address_line_1")
    rentcast_gdf = gpd.GeoDataFrame(
        rentcast_df,
        geometry=gpd.points_from_xy(
            x=rentcast_df["longitude"], y=rentcast_df["latitude"]
        ),
        crs="EPSG:4326",
    )
    rentcast_gdf = rentcast_gdf[
        ["rentcast_idx", "formattedAddress", "address_line_1", "geometry"]
    ]

    # Merge DataFrames
    merged_gdf = rentcast_gdf.merge(
        right=yelp_gdf,
        how="inner",
        on="address_line_1",
    )

    # Mark Yelp records lacking matches in Rentcast data
    all_yelp_indices = set(list(yelp_gdf.yelp_idx))
    merged_yelp_indices = set(list(merged_gdf.yelp_idx))
    missing_indices = all_yelp_indices.difference(merged_yelp_indices)
    yelp_gdf["matched"] = ~yelp_gdf.yelp_idx.isin(missing_indices)

    # Determine center point of apartments to determine initial map location
    combined_gdf = pd.concat([rentcast_gdf[["geometry"]], yelp_gdf[["geometry"]]])
    original_crs = combined_gdf.crs
    utm_crs = combined_gdf.estimate_utm_crs()
    center_lon, center_lat = (
        combined_gdf.to_crs(utm_crs)
        .dissolve()
        .centroid.to_crs(original_crs)
        .iloc[0]
        .coords[:][0]
    )

    # Plot points
    fmap = folium.Map(location=(center_lat, center_lon), zoom_start=12)
    for _, row in rentcast_gdf.iterrows():
        lon, lat = row.geometry.coords[:][0]
        popup = folium.Popup(html=row["address_line_1"])
        folium.CircleMarker(
            location=(lat, lon), radius=2, fillColor="blue", popup=popup
        ).add_to(fmap)

    for _, row in yelp_gdf.iterrows():
        lon, lat = row.geometry.coords[:][0]
        popup = folium.Popup(html=f"<b>{row['Subject']}</b><br/>{row['Location']}")
        folium.CircleMarker(
            location=(lat, lon),
            radius=2,
            color="orange" if row["matched"] else "red",
            fillColor="red",
            popup=popup,
        ).add_to(fmap)

    # Create map legend
    legend_html = """'
        {% macro html(this, kwargs) %}
        <div 
            style="
                background: white;
                padding: 5px;
                bottom: 50px;
                right: 10px;
                position: fixed;
                z-index: 9999;
            "
        >
            <p style="font-weight: bold;">Legend</p>
            <p>
                🔵- RentCast Apartment<br/>
                🟠- Matched Yelp Apartment<br/>
                🔴- Unmatched Yelp Apartment<br/>
            </p>
        </div>
        {% endmacro %}
        """
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    # folium.LayerControl().add_to(fmap)
    fmap.get_root().add_child(legend)

    # Render match statistics and map
    display(HTML("<b>Matches</b>"))
    display(
        f"{len(merged_yelp_indices)} of the {len(all_yelp_indices)} "
        "Yelp apartments had a match in the Rentcast dataset "
        f"({len(merged_yelp_indices) / len(all_yelp_indices):2f} percent)."
    )
    display(fmap)


def fetch_all_properties(
    city: str, state: str, property_type: str, api_key: str
) -> None:
    """Queries the Rentcast API for property data and then writes the data to file.
    To handle pagination in API calls, where we might need to make multiple
    requests to fetch all data, we can dynamically adjust the offset based on the
    number of items returned in the response. If a response returns fewer than the
    maximum limit (500 in this case), it implies there are no more records to fetch.

    Args:
        city (`str`): The name of the city to fetch.

        state (`str`): The state corresponding to the city.

        property_type (`str`): The type of property to fetch in the city
            (e.g., "Single Family" for single family dwelling).

        api_key (`str): The API key used for authentication.

    Returns:
        `None`
    """
    base_url = "https://api.rentcast.io/v1/properties"
    limit = 500
    offset = 0
    all_data = []

    while True:
        # Construct the URL with the current offset
        url = f"{base_url}?city={city}&state={state}&propertyType={property_type}&limit={limit}&offset={offset}"

        # Headers including the API key
        headers = {"accept": "application/json", "X-Api-Key": api_key}

        # Making the API call
        response = requests.get(url, headers=headers)
        data = response.json()

        # Check the number of properties returned
        num_properties = len(data)
        all_data.extend(data)

        # If fewer than limit properties are returned, stop fetching
        if num_properties < limit:
            break

        # Otherwise, increase the offset by the limit for the next call
        offset += limit

    # Save the collected data to a JSON file
    with open(
        f'{city}_{state}_{property_type.replace("-", "_")}_data.json', "w"
    ) as file:
        json.dump(all_data, file, indent=4)

    return f"Data fetched and saved for {city}, {state}, type: {property_type}"
