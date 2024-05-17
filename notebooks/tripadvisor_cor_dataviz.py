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
import math
import numpy as np
import usaddress
import re
from thefuzz import fuzz
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import logging
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
import os

### RELATIVE IMPORTS ###
from utils.config import ADDRESS_MAPPING
import tripadvisor_hotels_webcrawl as thw


file_dir = os.path.dirname(os.path.abspath("__file__"))

DATA_PATH = os.path.join(file_dir, "..", "data")
ENV_PATH = os.path.join(file_dir, "..", ".env")

INFO_PATH = os.path.join(DATA_PATH, "Infogroup", "2023_Business_Academic_QCQ.txt")

HOTEL_LR_DEPENDENT_VARIABLES = ["parent_sales_volume", "sales_volume", "employee_size"]
HOTEL_LR_INDEPENDENT_VARIABLES = ["number_of_rooms", "num_reviews", "rating", "price_level_category"]
HOTEL_LR_VARIABLES = HOTEL_LR_DEPENDENT_VARIABLES + HOTEL_LR_INDEPENDENT_VARIABLES


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
    # df.loc[:, ["infogroup"]] = True
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


def get_price_level_category(price_level):
    """_summary_

    Args:
        price_level (_type_): _description_

    Returns:
        _type_: _description_
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


def format_hotel_df(hotel_df: pd.DataFrame):
    hotel_df.loc[:, "price_level_category"] = hotel_df.loc[:, "price_level"].apply(lambda price_level: get_price_level_category(price_level))
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
    """ Plots the hotels, businesses, and foot traffic for the provided city.
    
    Args:
        cityname (`str`): The name of the city.
        city_geo (`Polygon`, `MultiPolygon`): The city boundary.
        hotel_business_visitor_df (`pd.DataFrame`): The hotel, business, and 
            foot traffic data.
        xlim_dim (`list`): The x-axis limits.
        ylim_dim (`list`): The y-axis limits.
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
    large_hotel_marker = Line2D([0], [0], marker="+", color="dodgerblue", label="Large Hotel (40 or more rooms)", linestyle="None")
    small_hotel_marker = Line2D([0], [0], marker=".", color="dodgerblue", label="Small Hotel (Fewer than 40 rooms)", linestyle="None")

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
        if row.get("number_of_rooms") >= 40:
            ax.text(row.geometry.x, row.geometry.y, "+", color="dodgerblue", fontsize=15, ha="center", va="center")
        else:
            ax.text(row.geometry.x, row.geometry.y, "o", color="dodgerblue", fontsize=11, ha="center", va="center")

    ctx.add_basemap(ax, crs=plot_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)


    ax.set_xlim([min_longitude, max_longitude])
    ax.set_ylim([min_latitude, max_latitude])

    ax.set_title(f"{cityname} Hotels, Businesses, and Foot Traffic")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(handles=[boundary_line, visitor_patch, sales_patch, large_hotel_marker, small_hotel_marker])

    plt.show()


def standardize_address(address: str, mappings = ADDRESS_MAPPING):
    """_summary_

    Args:
        address (str): _description_
        mappings (Dict, optional): _description_. Defaults to ADDRESS_MAPPING.

    Returns:
        _type_: _description_
    """
    address = address.upper().strip()
    parsed_address = usaddress.parse(address)
    standardized_address = [mappings.get(text, text) for text, _ in parsed_address] 
    return " ".join(standardized_address)


def combined_similarity(hotel_address, hotel_name, business_address, business_name, 
        address_weight=0.5, name_weight=0.5
):
    """_summary_

    Args:
        hotel (_type_): _description_
        business (_type_): _description_
        address_weight (float, optional): _description_. Defaults to 0.5.
        name_weight (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    address_score = fuzz.partial_ratio(hotel_address, business_address)
    name_score = fuzz.partial_ratio(hotel_name, business_name)
    combined_score = (address_weight * address_score) + (name_weight * name_score)
    return combined_score


def find_matching(hotel_df, business_df, hotel_name_column: str = "name", 
        hotel_address_column: str = "street1", 
        business_name_column: str = "name", 
        business_address_column: str = "street1",
        address_weight=0.5, name_weight=0.5, min_score=86
    ):
    """_summary_

    Args:
        hotel (_type_): _description_
        business_df (_type_): _description_
        address_weight (float, optional): _description_. Defaults to 0.5.
        name_weight (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    indicies_lst = []
    
    
    for hotel_index, hotel in hotel_df.iterrows():
        if pd.isna(hotel[hotel_address_column]):
            continue
        if pd.isna(hotel[hotel_name_column]):
            continue
        hotel[hotel_address_column] = str(hotel[hotel_address_column])
        for business_index, business in business_df.iterrows():
            # print(business, hotel)
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


def merge_data(hotel_df, business_df, indices_lst):
    """_summary_

    Args:
        hotel_df (_type_): _description_
        business_df (_type_): _description_
        indicies_lst (_type_): _description_

    Returns:
        _type_: _description_
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
    cityname = cityname.upper().strip()
    state_abbrev = state_abbrev.upper().strip()
    info_df = create_infogroup_df(INFO_PATH, cityname, state_abbrev)
    business_df = format_infogroup_df(info_df)

    return business_df


def get_city_foot_traffic(cityname: str):
    cityname = cityname.lower().replace(" ", "_").strip()
    foot_traffic_path = os.path.join(DATA_PATH, "foot-traffic", f"{cityname}_full_patterns.parquet")
    foot_traffic_df = pd.read_parquet(foot_traffic_path)
    foot_traffic_df = create_aggregated_foottraffic_df(foot_traffic_df)
    
    return foot_traffic_df


def get_city_hotels(cityname: str, api_or_crawled: str):
    cityname = cityname.lower().replace(" ", "_").strip()
    if api_or_crawled == "api":
        hotel_path = os.path.join(DATA_PATH, "tripadvisor", f"{cityname}_hotels_api.csv")
    elif api_or_crawled == "crawled":
        hotel_path = os.path.join(DATA_PATH, "tripadvisor", f"{cityname}_hotels_crawled.csv")
    hotel_df = pd.read_csv(hotel_path)
    hotel_df = format_hotel_df(hotel_df)
    
    return hotel_df


def get_city_geo(cityname: str):
    cityname = cityname.title().strip()
    city_geo = thw.CityGeo(cityname).geo
    
    return city_geo


def get_city_merged(hotel_df, business_df):
    city_indicies_lst = find_matching(hotel_df, business_df)
    city_merged_df, city_complete_df = merge_data(hotel_df, business_df, city_indicies_lst)
    
    return city_merged_df, city_complete_df


def get_city_data(cityname: str, state_abbrev: str):
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


def get_city_linear_regression(cityname, df, api_or_crawled):
    df_copy = df.loc[:, HOTEL_LR_VARIABLES].copy()
    df_copy.dropna(inplace=True)
    X = df.loc[:, HOTEL_LR_INDEPENDENT_VARIABLES]
    Y = df.loc[:, HOTEL_LR_DEPENDENT_VARIABLES]
    for dependent in HOTEL_LR_DEPENDENT_VARIABLES:
        print(f"_____{cityname.title()} {api_or_crawled.title()}: {dependent.replace('_', ' ').title()}_____""")
        y = df.loc[:, dependent]
        for i in range(2, len(HOTEL_LR_INDEPENDENT_VARIABLES) + 1):
            for combo in combinations(HOTEL_LR_INDEPENDENT_VARIABLES, i):
                X = df_copy.loc[:, combo]
                print(f"y: {y.head(1)}")
                print(f"X: {X.head(1)}")
                y, X = y.align(X, join='inner', axis=0)
                X = sm.add_constant(X)  # Adds a constant term to the predictor
                model = sm.OLS(y, X).fit()
                print(model.summary())


def get_city_correlations(cityname, df_key, df):
    plt.figure(figsize=(16, 6))
    df = df.loc[:, HOTEL_LR_VARIABLES].copy()
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title(
        f"Correlation Heatmap {cityname.title()} {df_key.replace('_', ' ').title()}", 
        fontdict={'fontsize':18}, pad=12
    )    
    heatmap.get_figure()
    
    return heatmap


def get_heatmaps(cityname, city_df_dict):
    api_heatmap = get_city_correlations(cityname, "api_merged", 
            city_df_dict["api_merged"]
    )
    crawled_heatmap = get_city_correlations(cityname, "crawled_merged", 
            city_df_dict["crawled_merged"]
    )
    
    return api_heatmap, crawled_heatmap


def get_city_linear_regression(cityname, df, api_or_crawled):
    df_copy = df.loc[:, HOTEL_LR_VARIABLES].copy()
    df_copy.dropna(inplace=True)
    X = df.loc[:, HOTEL_LR_INDEPENDENT_VARIABLES]
    Y = df.loc[:, HOTEL_LR_DEPENDENT_VARIABLES]
    for dependent in HOTEL_LR_DEPENDENT_VARIABLES:
        print(f"_____{cityname.title()} {api_or_crawled.title()}: {dependent.replace('_', ' ').title()}_____""")
        y = df.loc[:, dependent]
        for i in range(2, len(HOTEL_LR_INDEPENDENT_VARIABLES) + 1):
            for combo in combinations(HOTEL_LR_INDEPENDENT_VARIABLES, i):
                X = df_copy.loc[:, combo]
                print(f"y: {y.head(1)}")
                print(f"X: {X.head(1)}")
                y, X = y.align(X, join='inner', axis=0)
                X = sm.add_constant(X)  # Adds a constant term to the predictor
                model = sm.OLS(y, X).fit()
                print(model.summary())