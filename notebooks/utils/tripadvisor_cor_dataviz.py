import pandas as pd
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx
from typing import Union, Optional
from shapely.geometry import Polygon, MultiPolygon
from typing import Dict, Union, List
import numpy as np
import usaddress
from thefuzz import fuzz
import logging
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
import os
import warnings


### RELATIVE IMPORTS ###
from utils.constants import ADDRESS_MAPPING, HOTEL_LR_DEPENDENT_VARIABLES
from utils.constants import HOTEL_LR_INDEPENDENT_VARIABLES, HOTEL_LR_VARIABLES
from utils import city_boundary_creation as cbc


file_dir = os.path.dirname(os.path.abspath("__file__"))

DATA_PATH = os.path.join(file_dir, "..", "data")
ENV_PATH = os.path.join(file_dir, "..", ".env")

INFO_PATH = os.path.join(DATA_PATH, "Infogroup", "2023_Business_Academic_QCQ.txt")


logging.basicConfig(level=logging.INFO, filename="tripadvisor_cor_dataviz.log", filemode="w", 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
handler = logging.FileHandler("tripadvisor_cor_dataviz.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")



def create_infogroup_df(file_name: str, city: str, state: str):
    """ Reads in the infogroup data and filters it by city and state.
    
        Args:
            file_name (str): The name of the file.
            city (str): The city name.
            state (str): The state name.
    
        Returns:
            pd.DataFrame: The filtered infogroup data.
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
            # print(df.columns)
            df_filtered = df.loc[(df.loc[:, "CITY"] == city) & (
                    df.loc[:, "STATE"] == state), :]
            if len(df_filtered) > 0:
                whole_df = df_filtered if whole_df is None else pd.concat([whole_df, df_filtered])
            if not lines_left:
                return whole_df
            
            
def format_infogroup_df(df: pd.DataFrame):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df.loc[:, ["COMPANY", "CITY", "ADDRESS LINE 1", "LATITUDE", "LONGITUDE", "SALES VOLUME (9) - LOCATION", "EMPLOYEE SIZE (5) - LOCATION", "PARENT ACTUAL SALES VOLUME"]]
    df = df.rename(columns={"ADDRESS LINE 1": "street1", "CITY": "city", "COMPANY": "name", "LATITUDE": "latitude", "LONGITUDE": "longitude", "SALES VOLUME (9) - LOCATION": "sales_volume", "EMPLOYEE SIZE (5) - LOCATION": "employee_size", "PARENT ACTUAL SALES VOLUME": "parent_sales_volume"})
    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)
    
    return df


def create_aggregated_foottraffic_df(patterns_df: pd.DataFrame):
    """ 
    """
    patterns_df.loc[:, "city"] = patterns_df.loc[:, "city"].str.lower().str.strip()
    patterns_df.loc[:, "year"] = patterns_df.loc[:, "date_range_start"].str[0:4].astype("Int64")
    foot_df = patterns_df.loc[(patterns_df.loc[:, "year"] >= 2018), :]
    foot_df.loc[:, "location_name"] = foot_df.loc[:, "location_name"].str.upper().str.strip()
    foot_df.loc[:, "street_address"] = foot_df.loc[:, "street_address"].str.upper().str.strip()
    foot_traffic_df = foot_df.groupby(["location_name", "latitude", "longitude", "street_address"]).agg({"raw_visit_counts": "sum", "raw_visitor_counts": "sum"}).reset_index()
    
    return foot_traffic_df


def get_price_level_category(price_level: str):
    """Converts the price level to a categorical variable.

    Args:
        price_level (str): _description_

    Returns:
        int: Category of the price level.
    """
    if price_level == "$":
        return 0
    elif price_level == "$$":
        return 1
    elif price_level == "$$$":
        return 2
    elif price_level == "$$$$":
        return 3
    else:
        return np.nan
    

def get_binary_room_category(number_of_rooms: int):
    """Converts the number of rooms to a binary variable.

    Args:
        number_of_rooms (int): Number of rooms in a lodging.

    Returns:
        int: Category of the number of rooms.
    """
    if not number_of_rooms:
        return np.nan
    elif number_of_rooms >= 90:
        return 1
    elif number_of_rooms < 90:
        return 0
    else:
        return np.nan


def format_hotel_df(hotel_df: pd.DataFrame):
    """Formats the hotel dataframe.

    Args:
        hotel_df (pd.DataFrame): The dataframe of hotel data.  Each row 
            represents a hotel.

    Returns:
        pd.DataFrame: The formatted hotel dataframe.
    """
    hotel_df.loc[:, "price_level_category"] = hotel_df.loc[:, "price_level"].apply(lambda price_level: get_price_level_category(price_level))
    hotel_df.loc[:, "large_hotel"] = hotel_df.loc[:, "number_of_rooms"].apply(lambda number_of_rooms: get_binary_room_category(number_of_rooms))
    hotel_df.loc[:, "name"] = hotel_df.loc[:, "name"].str.upper().str.strip()
    hotel_df.loc[:, "street1"] = hotel_df.loc[:, "street1"].str.upper().str.strip()
    hotel_df.loc[:, "city"] = hotel_df.loc[:, "city"].str.upper().str.strip()
    
    return hotel_df


def plot_hotel_business_visitor(cityname: str, 
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
        cityname (str): The name of the city.
        city_geo (Union[Polygon, MultiPolygon]): The boundary of the city.
        hotel_business_visitor_df (pd.DataFrame)): The concated dataframe of 
            hotel, business, and foot-traffic data.
        min_longitude (float, optional): The min longitude for the plot. 
            Defaults to None.
        max_longitude (float, optional): The max longitude for the plot. 
            Defaults to None.
        min_latitude (float, optional): The min latitude for the plot. 
            Defaults to None.
        max_latitude (_type_, optional): The max latitude for the plot. 
                Defaults to None.
        scale_factor (float, optional): The scale of the buffers for the 
            business and foot-traffic data. Defaults to 0.0000001.
    """
    hotel_business_visitor_df["raw_visitor_counts"] = pd.to_numeric(
            hotel_business_visitor_df["raw_visitor_counts"], errors='coerce'
    )
    hotel_business_visitor_df["sales_volume"] = pd.to_numeric(
            hotel_business_visitor_df["sales_volume"], errors='coerce'
    )

    max_visitor_counts = int(hotel_business_visitor_df["raw_visitor_counts"].max())
    min_visitor_counts = int(hotel_business_visitor_df["raw_visitor_counts"].min())
    max_sales_volume = int(hotel_business_visitor_df["sales_volume"].max())
    min_sales_volume = int(hotel_business_visitor_df["sales_volume"].min())
    
    city_boundary = gpd.GeoDataFrame(geometry=[city_geo], crs="EPSG:4326")

    plot_gdf = gpd.GeoDataFrame(
        hotel_business_visitor_df,
        geometry=gpd.points_from_xy(hotel_business_visitor_df.longitude, hotel_business_visitor_df.latitude),
        crs="EPSG:4326"
    )
    plot_gdf = plot_gdf[plot_gdf.geometry.within(city_boundary.unary_union)]
    
    if not min_longitude:
        max_latitude = plot_gdf.geometry.y.max()
        min_latitude = plot_gdf.geometry.y.min()
        max_longitude = plot_gdf.geometry.x.max()
        min_longitude = plot_gdf.geometry.x.min()


    fig, ax = plt.subplots(figsize=(10, 10))
    boundary_plot = city_boundary.boundary.plot(ax=ax, color="black", linewidth=1, label=f"{cityname} Boundary")

    boundary_line = Line2D([0], [0], color="black", linewidth=1, label=f"{cityname} Boundary")
    visitor_patch = mpatches.Circle((0, 0), 0.1, color="red", alpha=0.3, label=f"Aggregated Raw Visitor Counts ({min_visitor_counts}-{max_visitor_counts})")
    sales_patch = mpatches.Circle((0, 0), 0.1, color="yellow", alpha=0.1, label=f"Sales Volume per Business ({min_sales_volume}-{max_sales_volume})")
    large_hotel_marker = Line2D([0], [0], marker="+", color="dodgerblue", label="Large Hotel (90 or more rooms)", linestyle="None")
    small_hotel_marker = Line2D([0], [0], marker=".", color="green", label="Small Hotel (Fewer than 90 rooms)", linestyle="None")

    visitor_scale_factor = scale_factor
    sales_scale_factor = scale_factor

    for _, row in plot_gdf.iterrows():
        try:
            ax.add_patch(plt.Circle(
                (row.geometry.x, row.geometry.y),
                row["raw_visitor_counts"] * visitor_scale_factor,
                color="red", alpha=0.3
            ))
        except TypeError as e:
            continue
        try :
            ax.add_patch(plt.Circle(
                (row.geometry.x, row.geometry.y),
                row["sales_volume"] * sales_scale_factor,
                color="yellow", alpha=0.1
            ))
        except TypeError as e:
            continue

    for _, row in plot_gdf.dropna(subset=["location_id"]).iterrows():
        if row.get("number_of_rooms") >= 90:
            ax.text(row.geometry.x, row.geometry.y, "+", color="dodgerblue", fontsize=17, ha="center", va="center", fontweight="bold")
        else:
            ax.scatter(row.geometry.x, row.geometry.y, color="green", s=15, zorder=3)

    ctx.add_basemap(ax, crs=plot_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)


    ax.set_xlim([min_longitude, max_longitude])
    ax.set_ylim([min_latitude, max_latitude])

    ax.set_title(f"{cityname} Hotels, Businesses, and Foot Traffic")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(handles=[boundary_line, visitor_patch, sales_patch, large_hotel_marker, small_hotel_marker])

    plt.show()


def standardize_address(address: str, mappings = ADDRESS_MAPPING):
    """Uses the usaddress library to parse and standardize an address.

    Args:
        address (str): An address to standardize.
        mappings (Dict, optional): A dictionary used to abbreviate words like 
            street, north, and road. Defaults to ADDRESS_MAPPING.

    Returns:
        str: The standardized address.
    """
    address = address.upper().strip()
    parsed_address = usaddress.parse(address)
    standardized_address = [mappings.get(text, text) for text, _ in parsed_address] 
    return " ".join(standardized_address)


def combined_similarity(hotel_address, hotel_name, business_address, business_name, 
        address_weight=0.5, name_weight=0.5
):
    """Finds the combined similarity score of the address and name of a hotel
    and a business.

    Args:
        hotel_address (str): The address of the hotel.
        hotel_name (str): The name of the hotel.
        business_address (str): The address of the business.
        business_name (str): The name of the business.
        address_weight (float, optional): The weight of the address in the 
            similarity score. Defaults to 0.5.
        name_weight (float, optional): The weight of the name in the similarity 
            score. Defaults to 0.5.

    Returns:
        float: The combined similarity score.
    """
    address_score = fuzz.partial_ratio(hotel_address, business_address)
    name_score = fuzz.partial_ratio(hotel_name, business_name)
    combined_score = (address_weight * address_score) + (name_weight * name_score)
    return combined_score


def find_matching(hotel_df: pd.DataFrame, business_df: pd.DataFrame, 
        hotel_name_column: str = "name", 
        hotel_address_column: str = "street1", 
        business_name_column: str = "name", 
        business_address_column: str = "street1",
        address_weight=0.5, name_weight=0.5, min_score=86
    ):
    """Matches hotels to businesses based on the similarity of their address
    and name.

    Args:
        hotel_df (pd.DataFrame): The dataframe of hotel data.
        business_df (pd.DataFrame): The dataframe of business data.
        hotel_name_column (str, optional): The column name of the hotel name.
            Defaults to "name".
        hotel_address_column (str, optional): The column name of the hotel
            address. Defaults to "street1".
        business_name_column (str, optional): The column name of the business
            name. Defaults to "name".
        business_address_column (str, optional): The column name of the business
            address. Defaults to "street1".
        address_weight (float, optional): The weight of the address in the
            similarity score. Defaults to 0.5.
        name_weight (float, optional): The weight of the name in the similarity
            score. Defaults to 0.5.
        min_score (int, optional): The minimum similarity score for a match.
            Defaults to 86.
        
        
    Returns:
        List: A list of tuples containing the indices of the matching hotels
            and businesses.
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
            business[business_address_column] = str(business[business_address_column])
            # print (hotel[hotel_address_column][0:3], business[business_address_column][0:3])
            if hotel[hotel_address_column][0:3] == business[business_address_column][0:3]:
                hotel["standardized_address"] = standardize_address(hotel[hotel_address_column])
                business["standardized_address"] = standardize_address(business[business_address_column])
                hotel["standardized_name"] = hotel[hotel_name_column]
                business["standardized_name"] = business[business_name_column]
                similarity_score = combined_similarity(
                        hotel["standardized_address"], 
                        hotel["standardized_name"],
                        business["standardized_address"],
                        business["standardized_name"], 
                        address_weight, name_weight
                )
                if similarity_score > min_score:
                    logger.info(f"Hotel: {hotel[hotel_name_column]}, {hotel[hotel_address_column]} Business: {business[business_name_column]}, {business[business_address_column]} Similarity: {similarity_score}") 
                    indicies_lst.append((hotel_index, business_index, similarity_score))
                    break
    
    return indicies_lst


def merge_data(hotel_df: pd.DataFrame, business_df: pd.DataFrame, 
        indices_lst: List
    ):
    """Merges the hotel and business dataframes based the indices list provided
    from the fuzzy matching.

    Args:
        hotel_df (pd.DataFrame): The dataframe of hotel data.
        business_df (pd.DataFrame): The dataframe of business data.
        indices_lst (List): A list of tuples containing the indices of the
            matching hotels and businesses.

    Returns:
        Tuple of pd.DataFrame: The dataframe of only merged rows and the 
            dataframe of all rows, merged or not.
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

    hotel_df_copy.drop([hotel_index for hotel_index, _, _ in indices_lst], inplace=True)
    business_df_copy.drop([business_index for _, business_index, _ in indices_lst], inplace=True)

    complete_df = pd.concat([merged_df, hotel_df, business_df], axis=0).reset_index(drop=True)

    return merged_df, complete_df


def get_city_infogroup(cityname: str, state_abbrev: str):
    """Gathers the infogroup data for a city.

    Args:
        cityname (str): The name of the city.
        state_abbrev (str): The abbreviation of the state of the city.

    Returns:
        pd.DataFrame: The infogroup data for the city.
    """
    cityname = cityname.upper().strip()
    state_abbrev = state_abbrev.upper().strip()
    info_df = create_infogroup_df(INFO_PATH, cityname, state_abbrev)
    business_df = format_infogroup_df(info_df)

    return business_df


def get_city_foot_traffic(cityname: str):
    """Gathers the foot traffic data for a city.

    Args:
        cityname (str): The name of the city.

    Returns:
        pd.DataFrame: The foot traffic data for the city.
    """
    cityname = cityname.lower().replace(" ", "_").strip()
    foot_traffic_path = os.path.join(DATA_PATH, "foot-traffic", f"{cityname}_full_patterns.parquet")
    foot_traffic_df = pd.read_parquet(foot_traffic_path)
    foot_traffic_df = create_aggregated_foottraffic_df(foot_traffic_df)
    
    return foot_traffic_df


def get_city_hotels(cityname: str, api_or_crawled: str):
    """Gathers the hotel data for a city.

    Args:
        cityname (str): The name of the city.
        api_or_crawled (str): The source of the hotel data.

    Returns:
        pd.DataFrame: The hotel data for the city.
    """
    cityname = cityname.lower().replace(" ", "_").strip()
    if api_or_crawled == "api":
        hotel_path = os.path.join(DATA_PATH, "tripadvisor", f"{cityname}_hotels_api.csv")
    elif api_or_crawled == "crawled":
        hotel_path = os.path.join(DATA_PATH, "tripadvisor", f"{cityname}_hotels_crawled.csv")
    hotel_df = pd.read_csv(hotel_path)
    hotel_df = format_hotel_df(hotel_df)
    
    return hotel_df


def get_city_geo(cityname: str):
    """Gathers the spatially enabled data for the boundary of a city.

    Args:
        cityname (str): The name of the city.

    Returns:
        Union[Polygon, MultiPolygon]: The boundary of the city.
    """
    cityname = cityname.title().strip()
    city_geo = cbc.CityGeo(cityname).geo
    
    return city_geo


def get_city_merged(hotel_df: pd.DataFrame, business_df: pd.DataFrame):
    """Impletments the fuzzy matching and merging of the hotel and business 
        data via partial ratios.

    Args:
        hotel_df (pd.DataFrame): The dataframe of hotel data.
        business_df (pd.DataFrame): The dataframe of business data.

    Returns:
        tuple of pd.DataFrame: The dataframe of only merged rows and the
            dataframe of all rows, merged or not.
    """
    city_indicies_lst = find_matching(hotel_df, business_df)
    city_merged_df, city_complete_df = merge_data(hotel_df, business_df, city_indicies_lst)
    
    return city_merged_df, city_complete_df


def get_city_data(cityname: str, state_abbrev: str):
    """Gathers all the data for a city in a single dictionary.

    Args:
        cityname (str): The name of the city.
        state_abbrev (str): The abbreviation of the state of the city.

    Returns:
        Dict of pd.DataFrame: A dictionary of all the dataframes for the city.
    """
    city_df_dict = {}
    city_df_dict["business"] = get_city_infogroup(cityname, state_abbrev)
    city_df_dict["foot_traffic"] = get_city_foot_traffic(cityname)
    city_df_dict["api_hotels"] = get_city_hotels(cityname, "api")
    city_df_dict["crawled_hotels"] = get_city_hotels(cityname, "crawled")
    city_df_dict["geo"] = get_city_geo(cityname)
    city_df_dict["api_merged"], city_df_dict["api_complete"] = get_city_merged(
            city_df_dict["api_hotels"], city_df_dict["business"]
    )
    city_df_dict["crawled_merged"], city_df_dict["crawled_complete"] = get_city_merged(
            city_df_dict["crawled_hotels"], city_df_dict["business"]
    )
    city_df_dict["api_concat"] = pd.concat([city_df_dict["api_hotels"], 
            city_df_dict["foot_traffic"], city_df_dict["business"]], 
            ignore_index=True
    )
    city_df_dict["crawled_concat"] = pd.concat([city_df_dict["crawled_hotels"], 
            city_df_dict["foot_traffic"], city_df_dict["business"]], 
            ignore_index=True
    )
    
    return city_df_dict


def get_city_correlations(cityname: str, df_key: str, df: pd.DataFrame):
    """Gathers the correlations of the hotel data for a city in a heatmap which 
        displays the correlations.

    Args:
        cityname (str): The name of the city.
        df_key (str): The key of the dataframe in the city dataframe dictionary.
        df (pd.DataFrame): The dataframe of hotel data.

    Returns:
        sns.heatmap: The heatmap of the correlations of the hotel and 
            business data.
    """
    key_title = df_key.replace("_", " ").title()
    plt.figure(figsize=(16, 6))
    df = df.loc[:, HOTEL_LR_VARIABLES].copy()
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title(
        f"Correlation Heatmap {cityname.title()} {key_title}", 
        fontdict={'fontsize':18}, pad=12
    )    
    heatmap.get_figure()
    
    return heatmap


def get_heatmaps(cityname: str, city_df_dict: Dict[str, pd.DataFrame]):
    """Gathers the correlations of the hotel data for a city in heatmaps and a
    returns a tuple of the heatmaps.

    Args:
        cityname (str): The name of the city.
        city_df_dict (Dict[str, pd.DataFrame]): A dictionary of the dataframes
            for the city.

    Returns:
        tuple of sns.heatmap: A tuple of the heatmaps of the correlations of the
            hotel and business data.
    """
    api_heatmap = get_city_correlations(cityname, "api_merged", 
            city_df_dict["api_merged"]
    )
    print(f"The API Dataframe has {city_df_dict['api_merged'].shape[0]} rows")
    crawled_heatmap = get_city_correlations(cityname, "crawled_merged", 
            city_df_dict["crawled_merged"]
    )
    print(f"The Crawled Dataframe has {city_df_dict['crawled_merged'].shape[0]} rows")
    
    return api_heatmap, crawled_heatmap


def get_city_linear_regression(cityname: str, df: pd.DataFrame, 
        api_or_crawled: str, combo_count: int = 4):
    """Calculates and displays the linear regression of the hotel data for a
    city.

    Args:
        cityname (str): The name of the city.
        df (pd.DataFrame): The dataframe of hotel data.
        api_or_crawled (str): The source of the hotel data.
        combo_count (int, optional): The number of combinations to calculate
    """
    df_copy = df.loc[:, HOTEL_LR_VARIABLES].copy()
    df_copy.dropna(inplace=True)
    HOTEL_INDEPENDENT_VARIABLES = [col for col in HOTEL_LR_INDEPENDENT_VARIABLES if col != "large_hotel"]
    X = df.loc[:, HOTEL_INDEPENDENT_VARIABLES]
    Y = df.loc[:, HOTEL_LR_DEPENDENT_VARIABLES]
    for dependent in HOTEL_LR_DEPENDENT_VARIABLES:
        print("#"*100)
        print(f"{cityname.title()} {api_or_crawled.title()}: {dependent.replace('_', ' ').title()}")
        print("#"*100)
        print("#"*100)
        y = df.loc[:, dependent]
        for i in range(combo_count, len(HOTEL_INDEPENDENT_VARIABLES) + 1):
            for combo in combinations(HOTEL_INDEPENDENT_VARIABLES, i):
                X = df_copy.loc[:, combo]
                print(f"y: {y.head(1)}")
                print(f"X: {X.head(1)}")
                y, X = y.align(X, join='inner', axis=0)
                X = sm.add_constant(X)
                model = sm.OLS(y, X).fit()
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")
                    model = sm.OLS(y, X).fit()
                    print(model.summary())
                    print("#"*100)  
                

def build_hotel_table(df_dict: dict[pd.DataFrame], city_lst: List, 
        api_or_crawled: str, table_column_names_lst: List):
    """Builds a table of hotel data for a list of cities.

    Args:
        df_dict (dict[pd.DataFrame]): A dictionary of the dataframes for the 
            cities.
        city_lst (List): A list of the city names.
        api_or_crawled (str): The source of the hotel data.
        table_column_names_lst (List): A list of the column names for the table.

    Returns:
        pd.DataFrame: The table of hotel data for the cities.
    """
    hotel_table_lst = []
    hotel_table_df = pd.DataFrame()
    
    for city in city_lst:
        city_key = city.lower().replace(" ", "_")
        city_dict = df_dict[f"{city_key}"]
        city_hotels_df = city_dict[api_or_crawled]
        
        city_num_hotels = city_hotels_df.shape[0]
        city_num_large_hotels = city_hotels_df.loc[(city_hotels_df.loc[:, "number_of_rooms"] >= 90), :].shape[0]
        city_num_small_hotels = city_hotels_df.loc[(city_hotels_df.loc[:, "number_of_rooms"] < 90), :].shape[0]
        city_num_null_hotels = city_hotels_df.loc[(city_hotels_df.loc[:, "number_of_rooms"].isnull()), :].shape[0]
        percentage_large_hotels = (city_num_large_hotels / city_num_hotels) * 100
        percentage_null_hotels = (city_num_null_hotels / city_num_hotels) * 100
        
        hotel_table_lst.append([city, city_num_hotels, city_num_large_hotels, city_num_small_hotels, city_num_null_hotels, percentage_large_hotels, percentage_null_hotels])
        
    hotel_table_df = pd.DataFrame(hotel_table_lst, columns=table_column_names_lst)
    
    numeric_columns_lst = [col for col in table_column_names_lst if col != "City"]
    styled_table = hotel_table_df.style.background_gradient(cmap="Greens", subset=numeric_columns_lst)
    display(styled_table)
    
    return hotel_table_df


def create_and_display_hotel_tables(df_dict: Dict, city_lst: List,
        api_table_column_names_lst: List, crawled_table_column_names_lst: List
    ):
    """Creates and displays the API and Crawled hotel tables for a list of cities.
    
    Args:
        df_dict (Dict[pd.DataFrame]): A dictionary of the dataframes for the 
            cities.
        city_lst (List): A list of the city names.
        api_table_column_names_lst (List): A list of the column names for the 
            API table.
        crawled_table_column_names_lst (List): A list of the column names for 
            the Crawled table.
    
    Returns:
        Tuple of pd.DataFrame: A tuple of the API and Crawled hotel tables.
    """
    # Create and display API hotel table
    api_hotel_table_df = build_hotel_table(df_dict, city_lst, "api_hotels", api_table_column_names_lst)
    
    # Create and display Crawled hotel table
    crawled_hotel_table_df = build_hotel_table(df_dict, city_lst, "crawled_hotels", crawled_table_column_names_lst)
    
    return api_hotel_table_df, crawled_hotel_table_df