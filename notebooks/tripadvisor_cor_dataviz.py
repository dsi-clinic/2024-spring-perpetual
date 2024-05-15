import pandas as pd
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import contextily as ctx
from typing import Union, Optional
from shapely.geometry import Polygon, MultiPolygon
from typing import Dict, Union
import math
import numpy as np
import usaddress
import re
from thefuzz import fuzz, process

### RELATIVE IMPORTS ###
from utils.config import ADDRESS_MAPPING

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


def address_lineone_format(df: pd.DataFrame, abbreviation_dict: Dict, 
            address_lineone_column: str):
    """

    Args:
        df (_type_): _description_
    """
    df.loc[:, address_lineone_column] = df.loc[:, address_lineone_column].str.upper().str.replace(".", " ").str.replace(r'\s+', ' ', regex=True).str.strip()
            
            
def change_infogroup_column_names(df: pd.DataFrame):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df.loc[:, ["infogroup"]] = True
    df = df.loc[:, ["COMPANY", "CITY", "ADDRESS LINE 1", "LATITUDE", "LONGITUDE", "SALES VOLUME (9) - LOCATION", "EMPLOYEE SIZE (5) - LOCATION", "PARENT ACTUAL SALES VOLUME"]]
    df = df.rename(columns={"ADDRESS LINE 1": "street1", "CITY": "city", "COMPANY": "name", "LATITUDE": "latitude", "LONGITUDE": "longitude", "SALES VOLUME (9) - LOCATION": "sales_volume", "EMPLOYEE SIZE (5) - LOCATION": "employee_size", "PARENT ACTUAL SALES VOLUME": "parent_sales_volume"})
    return df


def change_foottraffic_column_names(df: pd.DataFrame):
    """ 
    """
    df.loc[:, ["foottraffic"]] = True
    df = df.rename(columns={"location_name": "name", "street_address": "street1"})
    return df



def plot_hotel_business_visitor(cityname: str, 
                                city_geo: Union[Polygon, MultiPolygon], 
                                hotel_business_visitor_df: pd.DataFrame, 
                                maxmin_visitor_counts : str,
                                maxmin_sales_volume : str,
                                scale_factor: float = 0.0000001,
                                xlim_dim: Optional[list] = None, 
                                ylim_dim: Optional[list] = None
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
    city_boundary = gpd.GeoDataFrame(geometry=[city_geo.geo], crs="EPSG:4326")

    plot_gdf = gpd.GeoDataFrame(
        hotel_business_visitor_df,
        geometry=gpd.points_from_xy(hotel_business_visitor_df.longitude, hotel_business_visitor_df.latitude),
        crs="EPSG:4326"
    )
    plot_gdf = plot_gdf[plot_gdf.geometry.within(city_boundary.unary_union)]

    fig, ax = plt.subplots(figsize=(10, 10))
    boundary_plot = city_boundary.boundary.plot(ax=ax, color="black", linewidth=1, label=f"{cityname} Boundary")

    boundary_line = Line2D([0], [0], color="black", linewidth=1, label=f"{cityname} Boundary")
    visitor_patch = mpatches.Circle((0, 0), 0.1, color="red", alpha=0.3, label=f"Aggregated Raw Visitor Counts {maxmin_visitor_counts}")
    sales_patch = mpatches.Circle((0, 0), 0.1, color="yellow", alpha=0.1, label=f"Sales Volume per Business {maxmin_sales_volume}")
    large_hotel_marker = Line2D([0], [0], marker="+", color="dodgerblue", label="Large Hotel (40 or more rooms)", linestyle="None")
    small_hotel_marker = Line2D([0], [0], marker="o", color="dodgerblue", label="Small Hotel (Fewer than 40 rooms)", linestyle="None")

    visitor_scale_factor = scale_factor
    sales_scale_factor = scale_factor

    for _, row in plot_gdf.iterrows():
        ax.add_patch(plt.Circle(
            (row.geometry.x, row.geometry.y),
            row["raw_visitor_counts"] * visitor_scale_factor,
            color="red", alpha=0.3
        ))

    for _, row in plot_gdf.iterrows():
        ax.add_patch(plt.Circle(
            (row.geometry.x, row.geometry.y),
            row["sales_volume"] * sales_scale_factor,
            color="yellow", alpha=0.1
        ))

    for _, row in plot_gdf.dropna(subset=["location_id"]).iterrows():
        if row.get("number_of_rooms") >= 40:
            ax.text(row.geometry.x, row.geometry.y, "+", color="dodgerblue", fontsize=15, ha="center", va="center")
        else:
            ax.text(row.geometry.x, row.geometry.y, "o", color="dodgerblue", fontsize=11, ha="center", va="center")

    ctx.add_basemap(ax, crs=plot_gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)

    if xlim_dim:
        ax.set_xlim(xlim_dim)
    if ylim_dim:
        ax.set_ylim(ylim_dim)

    ax.set_title(f"{cityname} Hotels, Businesses, and Foot Traffic")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(handles=[boundary_line, visitor_patch, sales_patch, large_hotel_marker, small_hotel_marker])

    plt.show()
    
    
def haversine(lat1, lon1, lat2, lon2): # https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/#
    """_summary_

    Args:
        lat1 (_type_): _description_
        lon1 (_type_): _description_
        lat2 (_type_): _description_
        lon2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # distance between latitudes
    # and longitudes
    dist_lat = (lat2 - lat1) * math.pi / 180.0
    dist_long = (lon2 - lon1) * math.pi / 180.0
 
    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    # apply formulae
    a = (pow(math.sin(dist_lat / 2), 2) +
         pow(math.sin(dist_long / 2), 2) *
             math.cos(lat1) * math.cos(lat2));
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c


def find_nearest(df, business_lon, business_lat, hotel_long, hotel_lat):
    """_summary_

    Args:
        business_lon (_type_): _description_
        business_lat (_type_): _description_
        hotel_long (_type_): _description_
        hotel_lat (_type_): _description_

    Returns:
        _type_: _description_
    """
    distances = np.sqrt((hotel_long - business_lon)**2 + (hotel_lat - business_lat)**2)
    nearest_index = distances.idxmin()
    return df_hotels.loc[nearest_index, 'Occupancy_Rate'], distances.min()


def standardize_address(address: str, mappings: Dict = ADDRESS_MAPPING):
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


def combined_similarity(hotel, business, address_weight=0.5, name_weight=0.5):
    """_summary_

    Args:
        hotel (_type_): _description_
        business (_type_): _description_
        address_weight (float, optional): _description_. Defaults to 0.5.
        name_weight (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    address_score = fuzz.partial_ratio(hotel.loc[:,"standardized_address"], business.loc[:, "standardized_address"])
    name_score = fuzz.partial_ratio(hotel.loc[:, "standardized_name"], business.loc[:, "standardized_name"])
    combined_score = (address_weight * address_score) + (name_weight * name_score)
    return combined_score


def find_matching(hotel_df, hotel_name_column, hotel_address_column, 
        business_df, business_name_column, business_address_column,
        address_weight=0.5, name_weight=0.5
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
    for _, hotel in hotel_df.iterrows():
        hotel.loc[:, hotel_address_column] = str(hotel.loc[:, hotel_address_column])
        for _, business in business_df.iterrows():
            business.loc[:, business_address_column] = str(business.loc[:, business_address_column])
            if hotel.loc[:, hotel_address_column][0:3] == business.loc[:, business_address_column][0:3]:
                hotel.loc[:, "standardized_address"] = standardize_address(hotel.loc[:, hotel_address_column])
                business.loc[:, "standardized_address"] = standardize_address(business.loc[:, business_address_column])
                hotel.loc[:, "standardized_name"] = hotel.loc[:, hotel_name_column].upper().strip()
                business.loc[:, "standardized_name"] = business.loc[:, business_name_column].upper().strip()
                combined_similarity = combined_similarity(
                        hotel.loc[:, "standardized_address"], 
                        hotel.loc[:, "standardized_name"],
                        business.loc[:, "standardized_address"],
                        business.loc[:, "standardized_name"], 
                        address_weight, name_weight
                )
                if combined_similarity > 90:
                    print(f"Hotel: {hotel.loc[:, hotel_name_column]} Business: {business.loc[:, business_name_column]} Similarity: {combined_similarity}")
                    break
                
