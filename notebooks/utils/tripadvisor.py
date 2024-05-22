""" Classes and functions exclusively used for the analysis of Tripadvisor hotels.
"""

# Standard library imports
import warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import usaddress
from fuzzywuzzy import fuzz
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas._libs.missing import NAType
from shapely.geometry import MultiPolygon, Polygon

# Application imports
from .constants import BOUNDARIES_DIR, TRIPADVISOR_DIR
from .infogroup import load_infogroup_data
from .logger import LoggerFactory
from .safegraph import load_foot_traffic

# Define constants
ADDRESS_MAPPING = {
    "STREET": "ST",
    "ROAD": "RD",
    "AVENUE": "AVE",
    "BOULEVARD": "BLVD",
    "LANE": "LN",
    "DRIVE": "DR",
    "COURT": "CT",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "PLACE": "PL",
    "SQUARE": "SQ",
    "UNIT": "UNIT",
    "APARTMENT": "APT",
    "SUITE": "STE",
    "FLOOR": "FL",
    "BUILDING": "BLDG",
}
HOTEL_LR_DEPENDENT_VARIABLES = [
    "parent_sales_volume",
    "sales_volume",
    "employee_size",
]
HOTEL_LR_INDEPENDENT_VARIABLES = [
    "large_hotel",
    "number_of_rooms",
    "num_reviews",
    "price_level_category",
    "rating",
]
HOTEL_LR_VARIABLES = (
    HOTEL_LR_DEPENDENT_VARIABLES + HOTEL_LR_INDEPENDENT_VARIABLES
)

# Define logger
logger = LoggerFactory.get("TRIPADVISOR LOG")


def get_price_level_category(price_level: str) -> Union[int, NAType]:
    """Converts a Tripadvisor price level to a categorical variable
    (i.e., the numbers 1, 2, or 3, or `np.nan` if no level, or
    an unexpected level, is given).

    Args:
        price_level (`str`): The price level.

    Returns:
        (`int` | `np.nan`): The corresponding category.
    """
    if price_level == "$":
        return 0
    if price_level == "$$":
        return 1
    if price_level == "$$$":
        return 2
    if price_level == "$$$$":
        return 3
    return np.nan


def get_binary_room_category(number_of_rooms: int) -> Union[int, NAType]:
    """Converts the number of rooms to a binary
    variable represented by 0 if the number of rooms
    is less than 90, 1 if the number of rooms is
    greater than 90, and `np.nan` otherwise.

    Args:
        number_of_rooms (`int`): The number of rooms in a lodging.

    Returns:
        (`int` | `np.nan`): The room number category.
    """
    if not number_of_rooms:
        return np.nan
    elif number_of_rooms >= 90:
        return 1
    elif number_of_rooms < 90:
        return 0
    else:
        return np.nan


def format_hotel_df(hotel_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans values within a Tripadvisor hotel DataFrame.

    Args:
        hotel_df (`pd.DataFrame`): The DataFrame, where
            each row represents a hotel.

    Returns:
        (`pd.DataFrame`): The formatted DataFrame.
    """
    hotel_df.loc[:, "price_level_category"] = hotel_df.loc[
        :, "price_level"
    ].apply(get_price_level_category)
    hotel_df.loc[:, "large_hotel"] = hotel_df.loc[:, "number_of_rooms"].apply(
        get_binary_room_category
    )
    hotel_df.loc[:, "name"] = hotel_df.loc[:, "name"].str.upper().str.strip()
    hotel_df.loc[:, "street1"] = (
        hotel_df.loc[:, "street1"].str.upper().str.strip()
    )
    hotel_df.loc[:, "city"] = hotel_df.loc[:, "city"].str.upper().str.strip()

    return hotel_df


def plot_hotel_business_visitor(
    cityname: str,
    city_geo: Union[Polygon, MultiPolygon],
    hotel_business_visitor_df: pd.DataFrame,
    min_longitude: Optional[float] = None,
    max_longitude: Optional[float] = None,
    min_latitude: Optional[float] = None,
    max_latitude: Optional[float] = None,
    scale_factor: float = 0.0000001,
):
    """Plots the hotels, businesses, and foot traffic data on a map.

    Args:
        cityname (`str`): The name of the city.

        city_geo (`Polygon` | `MultiPolygon`): The boundary of the city.

        hotel_business_visitor_df (`pd.DataFrame`): The concatenated
            Dataframe of hotel, business, and foot traffic data.

        min_longitude (`float` | `None`): The min longitude for the plot.
            Defaults to `None`.

        max_longitude (`float` | `None`, optional): The max longitude for the plot.
            Defaults to `None`.

        min_latitude (`float` | `None`, optional): The min latitude for the plot.
            Defaults to `None`.

        max_latitude (`float` | `None`, optional): The max latitude for the plot.
            Defaults to `None`.

        scale_factor (`float`, optional): The scale of the buffers for the
            business and foot-traffic data. Defaults to 0.0000001.

    Returns:
        `None`
    """
    hotel_business_visitor_df["raw_visitor_counts"] = pd.to_numeric(
        hotel_business_visitor_df["raw_visitor_counts"], errors="coerce"
    )
    hotel_business_visitor_df["sales_volume"] = pd.to_numeric(
        hotel_business_visitor_df["sales_volume"], errors="coerce"
    )

    max_visitor_counts = int(
        hotel_business_visitor_df["raw_visitor_counts"].max()
    )
    min_visitor_counts = int(
        hotel_business_visitor_df["raw_visitor_counts"].min()
    )
    max_sales_volume = int(hotel_business_visitor_df["sales_volume"].max())
    min_sales_volume = int(hotel_business_visitor_df["sales_volume"].min())

    city_boundary = gpd.GeoDataFrame(geometry=[city_geo], crs="EPSG:4326")

    plot_gdf = gpd.GeoDataFrame(
        hotel_business_visitor_df,
        geometry=gpd.points_from_xy(
            hotel_business_visitor_df.longitude,
            hotel_business_visitor_df.latitude,
        ),
        crs="EPSG:4326",
    )
    plot_gdf = plot_gdf[plot_gdf.geometry.within(city_boundary.unary_union)]

    if not min_longitude:
        max_latitude = plot_gdf.geometry.y.max()
        min_latitude = plot_gdf.geometry.y.min()
        max_longitude = plot_gdf.geometry.x.max()
        min_longitude = plot_gdf.geometry.x.min()

    _, ax = plt.subplots(figsize=(10, 10))
    city_boundary.boundary.plot(
        ax=ax, color="black", linewidth=1, label=f"{cityname} Boundary"
    )

    boundary_line = Line2D(
        [0], [0], color="black", linewidth=1, label=f"{cityname} Boundary"
    )
    visitor_patch = mpatches.Circle(
        (0, 0),
        0.1,
        color="red",
        alpha=0.3,
        label=(
            "Aggregated Raw Visitor Counts"
            f" ({min_visitor_counts}-{max_visitor_counts})"
        ),
    )
    sales_patch = mpatches.Circle(
        (0, 0),
        0.1,
        color="yellow",
        alpha=0.1,
        label=(
            f"Sales Volume per Business ({min_sales_volume}-{max_sales_volume})"
        ),
    )
    large_hotel_marker = Line2D(
        [0],
        [0],
        marker="+",
        color="dodgerblue",
        label="Large Hotel (90 or more rooms)",
        linestyle="None",
    )
    small_hotel_marker = Line2D(
        [0],
        [0],
        marker=".",
        color="green",
        label="Small Hotel (Fewer than 90 rooms)",
        linestyle="None",
    )

    visitor_scale_factor = scale_factor
    sales_scale_factor = scale_factor

    for _, row in plot_gdf.iterrows():
        try:
            ax.add_patch(
                plt.Circle(
                    (row.geometry.x, row.geometry.y),
                    row["raw_visitor_counts"] * visitor_scale_factor,
                    color="red",
                    alpha=0.3,
                )
            )
        except TypeError:
            continue
        try:
            ax.add_patch(
                plt.Circle(
                    (row.geometry.x, row.geometry.y),
                    row["sales_volume"] * sales_scale_factor,
                    color="yellow",
                    alpha=0.1,
                )
            )
        except TypeError:
            continue

    for _, row in plot_gdf.dropna(subset=["location_id"]).iterrows():
        if row.get("number_of_rooms") >= 90:
            ax.text(
                row.geometry.x,
                row.geometry.y,
                "+",
                color="dodgerblue",
                fontsize=17,
                ha="center",
                va="center",
                fontweight="bold",
            )
        else:
            ax.scatter(
                row.geometry.x, row.geometry.y, color="green", s=15, zorder=3
            )

    ctx.add_basemap(
        ax, crs=plot_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron
    )

    ax.set_xlim([min_longitude, max_longitude])
    ax.set_ylim([min_latitude, max_latitude])

    ax.set_title(f"{cityname} Hotels, Businesses, and Foot Traffic")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(
        handles=[
            boundary_line,
            visitor_patch,
            sales_patch,
            large_hotel_marker,
            small_hotel_marker,
        ]
    )

    plt.show()


def standardize_address(
    address: str, mappings: Dict[str, str] = ADDRESS_MAPPING
) -> str:
    """Uses the usaddress library to parse and standardize an address.

    Args:
        address (`str`): The address to standardize.

        mappings (`dict`, optional): A dictionary used to abbreviate words like
            "street", "north", and "road". Defaults to ADDRESS_MAPPING.

    Returns:
        (`str`): The standardized address.
    """
    address = address.upper().strip()
    parsed_address = usaddress.parse(address)
    standardized_address = [
        mappings.get(text, text) for text, _ in parsed_address
    ]
    return " ".join(standardized_address)


def combined_similarity(
    hotel_address: str,
    hotel_name: str,
    business_address: str,
    business_name: str,
    address_weight: float = 0.5,
    name_weight: float = 0.5,
) -> float:
    """Finds the combined similarity score of the address
    and name of a Tripadvisor hotel and Infogroup business.

    Args:
        hotel_address (`str`): The address of the hotel.

        hotel_name (`str`): The name of the hotel.

        business_address (`str`): The address of the business.

        business_name (`str`): The name of the business.

        address_weight (`float`, optional): The weight of the address in
            the similarity score. Defaults to 0.5.

        name_weight (`float`, optional): The weight of the name
            in the similarity score. Defaults to 0.5.

    Returns:
        (`float`): The combined similarity score.
    """
    address_score = fuzz.partial_ratio(hotel_address, business_address)
    name_score = fuzz.partial_ratio(hotel_name, business_name)
    combined_score = (address_weight * address_score) + (
        name_weight * name_score
    )
    return combined_score


def match_hotels_to_businesses(
    hotel_df: pd.DataFrame,
    business_df: pd.DataFrame,
    hotel_name_column: str = "name",
    hotel_address_column: str = "street1",
    business_name_column: str = "name",
    business_address_column: str = "street1",
    address_weight: float = 0.5,
    name_weight: float = 0.5,
    min_score: int = 86,
):
    """Matches hotels to businesses based on the
    similarity of their addresses and names.

    Args:
        hotel_df (`pd.DataFrame`): The DataFrame of hotel data.

        business_df (`pd.DataFrame`): The DataFrame of business data.

        hotel_name_column (`str`, optional): The column name of the hotel name.
            Defaults to "name".

        hotel_address_column (`str`, optional): The column name of the hotel
            address. Defaults to "street1".

        business_name_column (`str`, optional): The column name of the business
            name. Defaults to "name".

        business_address_column (`str`, optional): The column name of the business
            address. Defaults to "street1".

        address_weight (`float`, optional): The weight of the address in the
            similarity score. Defaults to 0.5.

        name_weight (`float`, optional): The weight of the name in the similarity
            score. Defaults to 0.5.

        min_score (`int`, optional): The minimum similarity score for a match.
            Defaults to 86.

    Returns:
        (`list` of (`int`, `int`, `float`)): A list of three-item tuples containing
            components of a matched hotel and business: (1) the index of the
            hotel in its DataFrame, (2) the index of the business in its DataFrame,
            and (3) the composite similarity score of the entities' names and addresses.
    """
    indicies_lst = []

    for hotel_index, hotel in hotel_df.iterrows():
        if pd.isna(hotel[hotel_address_column]):
            continue
        if pd.isna(hotel[hotel_name_column]):
            continue
        hotel[hotel_address_column] = str(hotel[hotel_address_column])
        for business_index, business in business_df.iterrows():
            if pd.isna(business[business_address_column]):
                continue
            if pd.isna(business[business_name_column]):
                continue
            business[business_address_column] = str(
                business[business_address_column]
            )
            if (
                hotel[hotel_address_column][0:3]
                == business[business_address_column][0:3]
            ):
                hotel["standardized_address"] = standardize_address(
                    hotel[hotel_address_column]
                )
                business["standardized_address"] = standardize_address(
                    business[business_address_column]
                )
                hotel["standardized_name"] = hotel[hotel_name_column]
                business["standardized_name"] = business[business_name_column]
                similarity_score = combined_similarity(
                    hotel["standardized_address"],
                    hotel["standardized_name"],
                    business["standardized_address"],
                    business["standardized_name"],
                    address_weight,
                    name_weight,
                )
                if similarity_score > min_score:
                    logger.info(
                        f"Hotel: {hotel[hotel_name_column]}, "
                        f"{hotel[hotel_address_column]} Business: "
                        f"{business[business_name_column]}, "
                        f"{business[business_address_column]} "
                        f"Similarity: {similarity_score}"
                    )
                    indicies_lst.append(
                        (hotel_index, business_index, similarity_score)
                    )
                    break

    return indicies_lst


def merge_hotels_and_businesses(
    hotel_df: pd.DataFrame,
    business_df: pd.DataFrame,
    indices_lst: List[int, int, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merges the hotel and business DataFrames based the
    indices list provided from the fuzzy matching.

    Args:
        hotel_df (`pd.DataFrame`): The dataframe of hotel data.

        business_df (`pd.DataFrame`): The dataframe of business data.

        indices_lst (`list` of (`int`, `int`, `float`)): A list of
            three-item tuples containing a hotel index, business
            index, and similarity score for a hotel-business match.

    Returns:
        ((`pd.DataFrame`, `pd.DataFrame`)): A two-item tuple consisting
            of a DataFrame of only merged rows and a DataFrame of all
            rows, merged or not.
    """
    merged_rows = []
    hotel_df_copy = hotel_df.copy()
    business_df_copy = business_df.copy()

    for hotel_index, business_index, _ in indices_lst:
        hotel = hotel_df_copy.loc[hotel_index].copy()
        business = business_df_copy.loc[business_index].copy()

        for column in business.index:
            if column not in hotel.index:
                hotel[column] = business[column]
            elif pd.isna(hotel[column]) and not pd.isna(business[column]):
                hotel[column] = business[column]

        merged_rows.append(hotel)

    merged_df = pd.DataFrame(merged_rows)

    hotel_df_copy.drop(
        [hotel_index for hotel_index, _, _ in indices_lst], inplace=True
    )
    business_df_copy.drop(
        [business_index for _, business_index, _ in indices_lst], inplace=True
    )

    complete_df = pd.concat(
        [merged_df, hotel_df, business_df], axis=0
    ).reset_index(drop=True)

    return merged_df, complete_df


def load_tripadvisor_hotels(city: str, source_type: str) -> pd.DataFrame:
    """Loads and cleans pre-collected Tripadvisor hotel data for a city.

    Raises:
        `ValueError` if `source_type` does not equal "api" or "crawled".

    Args:
        city (`str`): The name of the city.

        source_type (`str`): The source of the hotel data.
            Must be one of "api" or "crawled".

    Returns:
        (`pd.DataFrame`): The cleaned hotel data for the city.
    """
    # Format city name
    city = city.lower().replace(" ", "_").strip()

    # Load file
    if source_type not in ("api", "crawled"):
        raise ValueError(
            'Source type must be "api" or "crawled", but a '
            f"value of {source_type} was received."
        )
    hotels_fpath = TRIPADVISOR_DIR / f"{city}_hotels_{source_type}.csv"
    hotel_df = pd.read_csv(hotels_fpath)

    # Clean records
    hotel_df = format_hotel_df(hotel_df)

    return hotel_df


def get_city_geo(city: str) -> Union[MultiPolygon, Polygon]:
    """Fetches a city's geographic boundary as a shapely `Polygon` or `MultiPolygon`.

    Args:
        city (`str`): The name of the city.

    Returns:
        (`MultiPolygon` | `Polygon`): The boundary.
    """
    city = city.title().strip()
    boundary_fpath = BOUNDARIES_DIR / f"{city}.geojson"
    return gpd.read_file(boundary_fpath).geometry.iloc[0]


def get_city_merged(
    hotel_df: pd.DataFrame, business_df: pd.DataFrame
) -> pd.DataFrame:
    """Implements the fuzzy matching and merging of
        the hotel and business data for a given
        city via string similarity partial ratios.

    Args:
        hotel_df (`pd.DataFrame`): The DataFrame of hotel data.

        business_df (`pd.DataFrame`): The DataFrame of business data.

    Returns:
        (`pd.DataFrame`, `pd.DataFrame`): The DataFrame of only
            merged rows and the DataFrame of all rows, merged or not.
    """
    city_indicies_lst = match_hotels_to_businesses(hotel_df, business_df)
    city_merged_df, city_complete_df = merge_hotels_and_businesses(
        hotel_df, business_df, city_indicies_lst
    )
    return city_merged_df, city_complete_df


def get_city_data(city: str, state: str) -> Dict[str, pd.DataFrame]:
    """Gathers all the data for a city in a single dictionary.

    Args:
        city (`str`): The name of the city.

        state (`str`): The abbreviation of the state of the city.

    Returns:
        (`dict`): A dictionary with strings as keys and DataFrames
            as values, representing the different DataFrames to use
            in an analysis.
    """
    city_df_dict = {}
    city_df_dict["business"] = load_infogroup_data(city, state)
    city_df_dict["foot_traffic"] = load_foot_traffic(city)
    city_df_dict["api_hotels"] = load_tripadvisor_hotels(city, "api")
    city_df_dict["crawled_hotels"] = load_tripadvisor_hotels(city, "crawled")
    city_df_dict["geo"] = get_city_geo(city)
    city_df_dict["api_merged"], city_df_dict["api_complete"] = get_city_merged(
        city_df_dict["api_hotels"], city_df_dict["business"]
    )
    (
        city_df_dict["crawled_merged"],
        city_df_dict["crawled_complete"],
    ) = get_city_merged(
        city_df_dict["crawled_hotels"], city_df_dict["business"]
    )
    city_df_dict["api_concat"] = pd.concat(
        [
            city_df_dict["api_hotels"],
            city_df_dict["foot_traffic"],
            city_df_dict["business"],
        ],
        ignore_index=True,
    )
    city_df_dict["crawled_concat"] = pd.concat(
        [
            city_df_dict["crawled_hotels"],
            city_df_dict["foot_traffic"],
            city_df_dict["business"],
        ],
        ignore_index=True,
    )
    return city_df_dict


def get_city_correlations(city: str, df_key: str, df: pd.DataFrame) -> Axes:
    """Creates a heatmap of correlations for hotel data fields.

    Args:
        city (`str`): The name of the city.

        df_key (`str`): The key of the DataFrame in the
            city DataFrame dictionary.

        df (`pd.DataFrame`): The hotel data.

    Returns:
        (`matplotlib.axes.Axes`): The heatmap of the
            correlations of the hotel and business data.
    """
    key_title = df_key.replace("_", " ").title()
    plt.figure(figsize=(16, 6))
    df = df.loc[:, HOTEL_LR_VARIABLES].copy()
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title(
        f"Correlation Heatmap {city.title()} {key_title}",
        fontdict={"fontsize": 18},
        pad=12,
    )
    heatmap.get_figure()

    return heatmap


def get_heatmaps(
    city: str, city_df_dict: Dict[str, pd.DataFrame]
) -> Tuple[Axes, Axes]:
    """Creates heatmaps of hotel data correlations for a city and then
    returns a tuple of the heatmaps.

    Args:
        city (`str`): The name of the city.

        city_df_dict (`dict`): A dictionary of the DataFrames
            for the city, with their names as keys.

    Returns:
        ((`matplotlib.axes.Axes`, `matplotlib.axes.Axes`)): A
            tuple of the heatmaps of the correlations of the
            hotel and business data.
    """
    api_heatmap = get_city_correlations(
        city, "api_merged", city_df_dict["api_merged"]
    )
    print(f"The API Dataframe has {city_df_dict['api_merged'].shape[0]} rows")
    crawled_heatmap = get_city_correlations(
        city, "crawled_merged", city_df_dict["crawled_merged"]
    )
    print(
        "The Crawled Dataframe has"
        f" {city_df_dict['crawled_merged'].shape[0]} rows"
    )

    return api_heatmap, crawled_heatmap


def get_city_linear_regression(
    city: str, df: pd.DataFrame, api_or_crawled: str, combo_count: int = 4
) -> None:
    """Calculates and displays a linear regression of hotel data for a city.

    Args:
        city (`str`): The name of the city.

        df (`pd.DataFrame`): The dataframe of hotel data.

        api_or_crawled (`str`): The source of the hotel data.

        combo_count (`int`, optional): The number of combinations to calculate.

    Returns:
        `None`
    """
    df_copy = df.loc[:, HOTEL_LR_VARIABLES].copy()
    df_copy.dropna(inplace=True)
    hotel_independent_vars = [
        col for col in HOTEL_LR_INDEPENDENT_VARIABLES if col != "large_hotel"
    ]
    temp_df = df.loc[:, hotel_independent_vars]
    for dependent in HOTEL_LR_DEPENDENT_VARIABLES:
        print("#" * 100)
        print(
            f"{city.title()} {api_or_crawled.title()}:"
            f" {dependent.replace('_', ' ').title()}"
        )
        print("#" * 100)
        print("#" * 100)
        y = df.loc[:, dependent]
        for i in range(combo_count, len(hotel_independent_vars) + 1):
            for combo in combinations(hotel_independent_vars, i):
                temp_df = df_copy.loc[:, combo]
                print(f"y: {y.head(1)}")
                print(f"X: {temp_df.head(1)}")
                y, temp_df = y.align(temp_df, join="inner", axis=0)
                temp_df = sm.add_constant(temp_df)
                model = sm.OLS(y, temp_df).fit()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="kurtosistest only valid for n>=20"
                    )
                    model = sm.OLS(y, temp_df).fit()
                    print(model.summary())
                    print("#" * 100)


def build_hotel_table(
    df_dict: Dict[str, pd.DataFrame],
    city_lst: List[str],
    source_type: str,
    table_column_names_lst: List[str],
) -> pd.DataFrame:
    """Builds a table of hotel data for a list of cities.

    Raises:
        `ValueError` if `source_type` does not equal "api" or "crawled".

    Args:
        df_dict (`dict`): A dictionary of the DataFrames for the cities.

        city_lst (`list` of `str`): A list of the city names.

        source_type (`str`): The source of the hotel data.
            One of "api" or "crawled".

        table_column_names_lst (`list` of `str`): A list of the
            column names for the table.

    Returns:
        (`pd.DataFrame`): The table of hotel data for the cities.
    """
    if source_type not in ("api", "crawled"):
        raise ValueError(
            'Source type must be "api" or "crawled", but a '
            f"value of {source_type} was received."
        )

    hotel_table_lst = []
    hotel_table_df = pd.DataFrame()

    for city in city_lst:
        city_key = city.lower().replace(" ", "_")
        city_dict = df_dict[f"{city_key}"]
        city_df = city_dict[source_type]

        num_hotels = city_df.shape[0]
        num_large_hotels = city_df.query("number_of_rooms >= 90").shape[0]
        num_small_hotels = city_df.query("number_of_rooms < 90").shape[0]
        num_null_hotels = city_df.query(
            "number_of_rooms != number_of_rooms"
        ).shape[0]
        percent_large_hotels = (num_large_hotels / num_hotels) * 100
        percent_null_hotels = (num_null_hotels / num_hotels) * 100

        hotel_table_lst.append(
            [
                city,
                num_hotels,
                num_large_hotels,
                num_small_hotels,
                num_null_hotels,
                percent_large_hotels,
                percent_null_hotels,
            ]
        )

    hotel_table_df = pd.DataFrame(
        hotel_table_lst, columns=table_column_names_lst
    )

    numeric_columns_lst = [
        col for col in table_column_names_lst if col != "City"
    ]
    styled_table = hotel_table_df.style.background_gradient(
        cmap="Greens", subset=numeric_columns_lst
    )
    display(styled_table)

    return hotel_table_df


def create_and_display_hotel_tables(
    df_dict: Dict[str, pd.DataFrame],
    city_lst: List[str],
    api_table_column_names_lst: List[str],
    crawled_table_column_names_lst: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates and displays the API and Crawled hotel tables for a list of cities.

    Args:
        df_dict (`dict`): A dictionary of the DataFrames for the cities.

        city_lst (`list` of `str`): A list of the city names.

        api_table_column_names_lst (`list` of `str`): A list of the
            column names for the API table.

        crawled_table_column_names_lst (`list` of `str`): A list of
            the column names for the crawled table.

    Returns:
        ((`pd.DataFrame`, `pd.DataFrame`)): A tuple of the API
            and Crawled hotel tables.
    """
    # Create and display API hotel table
    api_hotel_table_df = build_hotel_table(
        df_dict, city_lst, "api_hotels", api_table_column_names_lst
    )

    # Create and display Crawled hotel table
    crawled_hotel_table_df = build_hotel_table(
        df_dict, city_lst, "crawled_hotels", crawled_table_column_names_lst
    )

    return api_hotel_table_df, crawled_hotel_table_df
