from bs4 import BeautifulSoup as soup
from typing import Optional
import numpy as np
import pickle
import os
import requests
from decimal import Decimal
from typing import Dict, List, Tuple, Union
from dotenv import load_dotenv
import json
from pathlib import Path
import time 
import logging
from requests.exceptions import HTTPError
from shapely import MultiPolygon, Polygon, box, Point
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, filename='trip_hotels_api.log', filemode='w', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
handler = logging.FileHandler('trip_hotels_api.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


### Custom imports ###
from common.geometry import BoundingBox, convert_meters_to_degrees
from common.logger import LoggerFactory
from foodware.places.common import PlacesSearchResult

### Constants ###
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH)
TRIPADVISOR_API_KEY = os.getenv("TRIPADVISOR_API_KEY")
TRIPADVISOR_API_KEY_SEC = os.getenv("TRIPADVISOR_API_KEY_SEC")
TRIPADVISOR_API_KEY_THIRD = os.getenv("TRIPADVISOR_API_KEY_THIRD")  
SP_WEBCRAWL_USER = os.getenv("SP_WEBCRAWL_USER")
SP_WEBCRAWL_PAS = os.getenv("SP_WEBCRAWL_PAS")
SP_RESI_USER = os.getenv("SP_RESI_USER")
SP_RESI_PAS = os.getenv("SP_RESI_PAS")
PROXY_HOST = "us.smartproxy.com"
PROXY_PORT = "10000"
TRIP_HOTELLST_URL = "https://www.tripadvisor.com/Hotels-g"
API_TRIP_LOC_SEARCH = "https://api.content.tripadvisor.com/api/v1/location/search?"
API_TRIP_NEARBY_SEARCH = "https://api.content.tripadvisor.com/api/v1/location/nearby_search?"
TRIP_HOTEL_SEARCH = "https://www.tripadvisor.com/Hotel_Review-g"
BASE_URL = "https://www.tripadvisor.com"
MAX_NUM_RESULTS_PER_REQUEST = 10
MAX_SEARCH_RADIUS_METERS: float = 50_000
MAX_SEARCH_RADIUS_MILES: float = 31.0686
SEARCH_RADIUS_UNIT: str = "mi" # mi is miles, m is meters
SECONDS_DELAY_PER_REQUEST: float = 0.5
BOUNDARIES_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "boundaries"
MAX_API_CALLS = 10
TRIP_NEARBY_LOC_KEYS = ["location_id", "name", "distance"] # also need dict["address_obj"]["address_string"] and dict["address_obj"]["city"]
TRIP_LOC_RESPONSE_KEYS = ["web_url", "latitude", "longitude", "rating", "num_reviews", "price_level"]


### Global Variables ###
global_location_id_lst = []
global_hotel_lst = []
global_filtered_hotel_lst = []


### Classes ###
class TripadvisorCityAPI:
    """
    """
    def __init__(self, cityname,
                boundaries_path = BOUNDARIES_DATA_PATH
    ):
        self.cityname = cityname
        self.cityname_geojson = self.prep_cityname_geojson(cityname)
        self.city_geojson_pathway = boundaries_path / f"{self.cityname_geojson}.geojson"
        self.city_geojson = self.load_boundary_geojson()
        self.geojson_type = self.city_geojson['features'][0]['geometry']['type'].lower().strip()
        self.geojson_coordinates = self.city_geojson['features'][0]['geometry']['coordinates']
        self.geo = self.city_geo()
        # self._api_key = TRIPADVISOR_API_KEY
        # self._api_key = TRIPADVISOR_API_KEY_SEC
        self._api_key = TRIPADVISOR_API_KEY_THIRD
        self._logger = LoggerFactory.get(__name__)
        self._max_num_results_per_request = MAX_NUM_RESULTS_PER_REQUEST
    
    
    def prep_cityname_geojson(self, cityname: str):
        """Prepares string representing the city's name for the geojson file by 
        removing whitespace, replacing spaces with _, and converting to lowercase.
        
        Args:
            cityname (`str`): The name of the city to be prepared.

        Returns:
            cityname (`str`): The prepared city name.
        """
        cityname = cityname.strip()
        cityname = cityname.replace(" ", "_")
        cityname = cityname.lower()
        
        return cityname


    def load_boundary_geojson(self):
        """Loads the geojson file for the city.
        
        Args:
            city_geojson_pathway (`str`): The pathway to the geojson file for the city.

        Returns:
            city_geojson (`dict`): The geojson file for the city.
        """
        with open(self.city_geojson_pathway, 'r') as file:
            city_geojson = json.load(file)
        
        return city_geojson
    
    
    def build_tripadvisor_nearby_api(self, lat_long: str, radius: int, radius_unit: str):
        """This method builds the URL for the Tripadvisor API call to search for
        hotels near a given latitude and longitude.

        Args:
            lat_long (`str`): The latitude and longitude of the location.
            radius (`int`): The radius to search for hotels, the unit should be
                in miles as they API has a bug with meters.
            radius_unit (`str`): The unit of the radius.
        Returns:
            api_call_params (`str`): The api call in string format.
        """
        api_call_params = f"latLong={lat_long}&key={self._api_key}&category=hotels&radius={radius}&radiusUnit={radius_unit}&language=en"
        return API_TRIP_NEARBY_SEARCH + api_call_params


    def hotel_api_call(self, hotel_location_id: int):
        """Calls the Tripadvisor API to search for the city and returns the JSON 
        response.
        
        Args: hotel_location_id (`int`): The location id of the hotel.
        
        Returns: response.json() (`dict`): The JSON response from the API call.
        """
        url = f"https://api.content.tripadvisor.com/api/v1/location/{hotel_location_id}/details?key={self._api_key}&language=en&currency=USD"
        headers = {"accept": "application/json"}
        try:
            response = requests.get(url, headers=headers)
        except Exception as e:
            self._logger.error(f"Error: {e}. Hotel_location_id: {hotel_location_id}")   
            return None
        if response.status_code == 403:
            if not self._api_key:
                raise ValueError("API key is missing.")
            else:
                raise HTTPError(f"{r.status_code}: The Tripadvisor API key is likely missing this device's IP address. Head to https://www.tripadvisor.com/developers?screen=credentials to add this device's IP address to the API key.")
        if response.status_code == 429:
            raise HTTPError(f"{response.status_code}: The Tripadvisor API has reached its limit of requests. Please try again later.")
        return response.json() 


    def api_hotel_info(self, response_hotel_lst: List[Dict]):
        """Takes the cleaned incomplete hotel dictionary list from 
        the clean_find_places_in_city() method and calls the hotel API to gather
        additional information on the hotels, such as the web_url, rating, and
        number of reviews.
        
        Args: 
            response_hotel_lst (`list` of `dict`): The data from the json 
                response of the Tripadvisor Nearby Location Search API call.  
                This is a list of hotel dictionaries with incomplete information.
        
        Returns:
            deduped_hotel_lst (`list` of `dict`): The data from the json 
                response of the Tripadvisor Nearby Location Search API call.  
                Combined with the data from the Tripadvisor Location Details
                API call for hotels within the city's boundaries.
        
        """
        deduped_hotel_lst = []
    
        for hotel_dict in response_hotel_lst:
            global global_location_id_lst
            if hotel_dict["location_id"] in global_location_id_lst:
                continue
            hotel_json = self.hotel_api_call(hotel_dict["location_id"])
            hotel_point = Point(hotel_json["longitude"], hotel_json["latitude"])
            nearby_loc_dict = {k: hotel_dict.get(k, None) for k in TRIP_NEARBY_LOC_KEYS}
            nearby_loc_dict["address_string"] = hotel_json["address_obj"]["address_string"]
            nearby_loc_dict["city"] = hotel_json["address_obj"]["city"]
            if "street1" in hotel_json["address_obj"]: 
                nearby_loc_dict["street1"] = hotel_json["address_obj"]["street1"]
            else:
                nearby_loc_dict["street1"] = np.nan
            loc_dict = {k: hotel_json.get(k, None) for k in TRIP_LOC_RESPONSE_KEYS}
            combined_hotel_dict = {**nearby_loc_dict, **loc_dict}
            global global_hotel_lst
            global_hotel_lst.append(combined_hotel_dict)
            if self.geo.contains(hotel_point):
                deduped_hotel_lst.append(combined_hotel_dict)
                global global_filtered_hotel_lst
                global_filtered_hotel_lst.append(combined_hotel_dict)   

            global_location_id_lst.append(hotel_dict["location_id"])

        return deduped_hotel_lst


    def city_geo(self):
        """This method creates a shapely polygon or multipolygon object from 
        the coordinates in the geojson file for the city.

        Returns:
            polygon_s (`shapely.Polygon` or `shapely.MultiPolygon`): The 
                polygon or multipolygon object for the city.
        """
        if self.geojson_type == "polygon":
            external_poly = self.geojson_coordinates[0]
            if len(self.geojson_coordinates) == 1:
                internal_polys = None
            else:
                internal_polys = self.geojson_coordinates[1:]
            polygon_s = Polygon(external_poly, internal_polys)
            
        elif self.geojson_type == "multipolygon":
            poly_lst = []
            for polygons in self.geojson_coordinates:
                external_poly = polygons[0]
                if len(polygons) == 1:
                    internal_polys = None
                else:
                    internal_polys = polygons[1:]
                poly_lst.append(Polygon(external_poly, internal_polys))
            polygon_s = MultiPolygon(poly_lst)
            
        else:
            raise ValueError("Invalid geometry type")
        
        return polygon_s


    def find_places_in_bounding_box(
        self, box: BoundingBox, search_radius: float
    ) -> Tuple[List[Dict], List[Dict]]:
        """Locates all POIs within the given area and categories.
        The area is further divided into a grid of quadrants if
        more results are available within the area than can be
        returned due to API limits.

        Args:
            box (`BoundingBox`): The bounding box.

            categories (`list` of `str`): The categories to search by.

            search_radius (`float`): The search radius, converted from
                meters to the larger of degrees longitude and latitude.

        Returns:
            ((`list` of `dict`, `list` of `dict`,)): A two-item tuple
                consisting of the list of retrieved places and a list
                of any errors that occurred, respectively.
        """
        
        api_params = {"latLong": f"{float(box.center.lat),float(box.center.lon)}",
                      "radius": search_radius,
                      "radiusUnit": SEARCH_RADIUS_UNIT
        }
        api_call_params = f"latLong={float(box.center.lat)},{float(box.center.lon)}&key={self._api_key}&category=hotels&radius={search_radius}&radiusUnit={SEARCH_RADIUS_UNIT}&language=en"

        # Send request to the Tripadvisor API
        def send_request():
            url = API_TRIP_NEARBY_SEARCH + api_call_params
            headers = {"accept": "application/json"}
            
            return requests.get(url, headers=headers)

        def retry(func, max_retries: int = 3):
            retries = 0
            last_error = ValueError("Max retries must be >= 0.")
            while retries <= max_retries:
                try:
                    return func()
                except Exception as e:
                    self._logger.error(f"Request failed. {e}")
                    retries += 1
                    last_error = e
                    time.sleep(3)
            raise last_error

        r = retry(send_request)

        if r.status_code == 403:
            if not self._api_key:
                raise ValueError("API key is missing.")
            else:
                raise HTTPError(f"{r.status_code}: The Tripadvisor API key is likely missing this device's IP address. Head to https://www.tripadvisor.com/developers?screen=credentials to add this device's IP address to the API key.")
        
        if r.status_code == 429:
            raise HTTPError(f"{r.status_code}: The Tripadvisor API has reached its limit of requests. Please try again later.")
        
        # Sleep and then parse JSON from response body
        try:
            time.sleep(SECONDS_DELAY_PER_REQUEST)
            data = r.json()["data"]
        except Exception as e:
            self._logger.error(f"Failed to parse reponse body JSON. {e}")
            return [], [], [{"api_params": api_params, "error": str(e)}]

        # If error occurred, store information and exit processing for cell
        if not r.ok or "error" in data:
            self._logger.error(
                "Failed to retrieve POI data through the Google Places API. "
                f'Received a "{r.status_code}-{r.reason}" status code '
                f'with the message "{r.text}".'
            )
            return [], [], [{"api_params": api_params, "error": data}]

        # Otherwise, if no data returned, return empty lists of POIs and errors
        if not data:
            self._logger.warning("No data found in response body.")
            return [], [], []
        deduped_filtered_data = self.api_hotel_info(data)
        
        # Plot the bounding box and their centers
        ############################################################    
        # plot = [box.to_shapely()]
        # plot.append(self.geo)
        # plot.append(Point(box.center.lon, box.center.lat))
        # plot_gdf = gpd.GeoDataFrame(geometry=plot)
        # plot_gdf.plot(facecolor="none", edgecolor="black")
        # plt.savefig("plot.png")
        ############################################################
        
        # Otherwise, if number of POIs returned equals max,
        # split box and recursively issue HTTP requests
        if len(data) >= MAX_NUM_RESULTS_PER_REQUEST:
            pois, pois_deduped, errors = [], [], []
            sub_cells = box.split_along_axes(x_into=2, y_into=2)
            
            # Plot the sub-cells and their centers
            ##############################################################    
            # plot = [cell.to_shapely() for cell in sub_cells]
            # plot.append(box.to_shapely())
            # plot.append(self.geo)
            # plot.append(Point(box.center.lon, box.center.lat))
            # plot.extend([Point(cell.center.lon, cell.center.lat) for cell in sub_cells])
            # # for cell in cells:
            # #     plot.append(cell.to_shapely())
            # plot_gdf = gpd.GeoDataFrame(geometry=plot)
            # plot_gdf.plot(facecolor="none", edgecolor="black")
            # plt.savefig("plot.png")
            ##############################################################
            
            for sub in sub_cells:
                if sub.intersects_with(self.geo):
                    sub_hotels, sub_deduped_hotels, sub_errs = self.find_places_in_bounding_box(
                        sub, search_radius / 2
                    )
                    pois.extend(sub_hotels)
                    pois_deduped.extend(sub_deduped_hotels)
                    errors.extend(sub_errs)
            return pois, pois_deduped, errors

        return data, deduped_filtered_data, []


    def find_places_in_geography(
        self, geo: Union[Polygon, MultiPolygon]
    ) -> Tuple[List[Dict], List[Dict]]:
        """NOTE: The POIs pulled can be limited as the API shows only the most 
        popular locations.
        
        Locates all POIs with a review within the given geography.
        The Tripadvisor API permits searching for POIs within a radius around
        a given point. The data is extracted by dividing the
        geography's bounding box into cells of equal size and then searching
        within the circular areas that circumscribe (i.e., perfectly enclose)
        those cells.

        To circumscribe a cell, the circle must have a radius that is
        one-half the length of the cell's diagonal (as derived from the
        Pythagorean Theorem). Let `side` equal the length of a cell's side.
        It follows that the radius is:

        ```
        radius = (√2/2) * side
        ```

        Tripadvisor sets a cap on the radius search size, so after solving for `side`,
        it follows that cell sizes are restricted as follows:

        ```
        max_side = √2 * max_radius
        ```

        Therefore, the bounding box must be split into _at least_ the following
        number of cells along the x- and y- (i.e., longitude and latitude)
        directions to avoid having cells that are too big:

        ```
        min_num_splits = ceil(bounding_box_length / max_side)
        ```

        Finally, at the time of writing, only 10 records are returned per
        search query, even if more businesses are available. Therefore, it
        is important to confirm that less than 10 records are returned
        in the response to avoid missing data.

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`PlacesResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)
        
        # # Calculate length of square circumscribed by circle with the max search radius
        box_side_meters = (2**0.5) * MAX_SEARCH_RADIUS_METERS

        # # Use heuristic to convert length from meters to degrees at box's lower latitude
        deg_lat, deg_lon = convert_meters_to_degrees(box_side_meters, bbox.bottom_left)

        # # Take minimum value as side length (meters convert differently to
        # # lat and lon, and we want to avoid going over max radius)
        box_side_degrees = min(deg_lat, deg_lon)

        # Divide box into grid of cells of approximately equal length and width
        # NOTE: Small size differences may exist due to rounding.
        cells: List[BoundingBox] = bbox.split_into_squares(
            size_in_degrees=Decimal(str(box_side_degrees))
        )

        # Locate POIs within each cell if it contains any part of geography
        hotel_lst, deduped_hotel_lst, errors = [], [], []
        
        # for batch in category_batches:
        for cell in cells:
            if cell.intersects_with(geo):
                cell_pois, cell_deduped_pois, cell_errors = self.find_places_in_bounding_box(
                    box=cell,
                    search_radius=MAX_SEARCH_RADIUS_MILES,
                )
                hotel_lst.extend(cell_pois)
                deduped_hotel_lst.extend(cell_deduped_pois)
                errors.extend(cell_errors)

        return PlacesSearchResult(hotel_lst, deduped_hotel_lst, errors)

    
    def global_hotel_lists(self):
        """This method returns the global hotel lists.

        Returns:
            (`PlacesResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        global global_hotel_lst
        global global_filtered_hotel_lst
        
        return PlacesSearchResult(global_hotel_lst, global_filtered_hotel_lst, [])


    def smartproxy_web_crawl(self, url: str, save_path: Optional[str] = None):
        """Uses Smartproxy Web Scrape to scrape a webpage and returns the page source 
        code.
        
        Args: 
            url (`str`): The url of the webpage to scrape.
            save_path (`str`): The pathway to save the page source code.
            
        Returns:
            page_source_code (`BeautifulSoup`): The page source code of the webpage.
        """
        task_params = {
            "target": "universal",
            "url": url # hotel_tripadvisor_url
            }

        response = requests.post(
            "https://scrape.smartproxy.com/v1/tasks",
            json = task_params,
            auth=(SP_WEBCRAWL_USER, SP_WEBCRAWL_PAS)
        )
        
        html_string = response.json()["results"][0]["content"]
        page_source_code  = soup(html_string, "lxml")
        if save_path:
            save_path = save_path + ".pkl"
            with open(save_path, "wb") as f:
                pickle.dump(page_source_code, f)
        
        return page_source_code


    def get_room_number(self, hotel_lst: List[Dict]):
        """Scrape the the hotel address and number of rooms 
        from each hotel's tripadvisor page.
        
        Args: 
            hotel_lst (`list` of `dict`): The list of hotel dictionaries.
        
        Returns: 
            hotel_lst (`list` of `dict`): The list of hotel dictionaries
                with the number of rooms added.
        """
        for index, hotel_dict in enumerate(hotel_lst):
            hotel_url = hotel_dict["web_url"]
            hotel_source_html = self.smartproxy_web_crawl(hotel_url, 
                    save_path = False
            )
            try:
                label_div = hotel_source_html.find("div", text="NUMBER OF ROOMS")
                number_of_rooms = label_div.find_next("div").text.strip() if label_div else np.nan
            except Exception as e:
                number_of_rooms = np.nan
                self._logger.error(f"Error: {e}.  Hotel_url: {hotel_url}")
            hotel_dict["number_of_rooms"] = number_of_rooms
            
            hotel_lst[index] = hotel_dict
        
        return hotel_lst


if __name__ == "__main__":
    Sunnyvale = TripadvisorCityAPI("Sunnyvale")
    Sunnyvale.find_places_in_geography(Sunnyvale.geo)
    Sunnyvale_api_database = Sunnyvale.global_hotel_lists()
    Sunnyvale_hotel_lst = Sunnyvale.get_room_number(Sunnyvale_api_database.clean)
    Sunnyvale_api_df = pd.DataFrame(Sunnyvale_hotel_lst)
    Sunnyvale_api_df = Sunnyvale_api_df.loc[(Sunnyvale_api_df.loc[:, "city"] == "Sunnyvale"), :]
    Sunnyvale_api_df.to_csv("Sunnyvale_hotels_api.csv", index=False)