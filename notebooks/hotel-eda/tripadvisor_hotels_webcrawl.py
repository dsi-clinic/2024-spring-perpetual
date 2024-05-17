from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as soup
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import os
import random
import requests
from requests.exceptions import HTTPError
from typing import Dict, List
from dotenv import load_dotenv
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import json
from pathlib import Path
import logging
import re
from shapely import MultiPolygon, Polygon

logging.basicConfig(level=logging.INFO, filename='trip_hotels_api.log', filemode='w', 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
handler = logging.FileHandler('trip_hotels_webcrawl.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


### Custom imports ###
from common.logger import LoggerFactory
from Smartproxy_residential.extension import proxies


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
TRIP_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tripadvisor_data"
BOUNDARIES_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "boundaries"
TRIP_CRAWlED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tripadvisor_data" / "crawled_data"
TRIP_LOC_RESPONSE_KEYS = ["location_id", "name", "web_url", "latitude", "longitude", "rating", "num_reviews", "price_level"] # also need dict["address_obj"]["address_string"] and dict["address_obj"]["city"]


class TripadvisorCityCrawl:
    def __init__(self, cityname: str, trip_data_path: Path = TRIP_DATA_PATH,
                boundaries_data_path: Path = BOUNDARIES_DATA_PATH,
                trip_crawled_data_path: Path = TRIP_CRAWlED_DATA_PATH
        ):
        self.cityname = cityname
        self.city_info = self.tripadvisor_city_info(cityname)
        self.initial_hotellst_url = self.tripadvisor_city_hotels()
        self._api_key = TRIPADVISOR_API_KEY
        # self._api_key = TRIPADVISOR_API_KEY_SEC
        # self._api_key = TRIPADVISOR_API_KEY_THIRD
        self._logger = LoggerFactory.get(__name__)  
        self.trip_data_path = trip_data_path
        self.boundaries_data_path = boundaries_data_path
        self.crawled_data_path = trip_crawled_data_path
        self.noroom_hotel_lst = self.crawl_hotels_lst()    
        self.hotel_lst = self.get_room_number()   
        self.hotel_df = pd.DataFrame(self.hotel_lst)  
        
                 
    def prep_cityname_api(self, cityname: str):
        '''Prepares string representing the city's name for API call by removing 
        whitespace, replacing spaces with %20, and converting to lowercase.
        
        Args:
            cityname (`str`): The name of the city to be prepared.

        Returns:
            cityname (`str`): The prepared city name.
        '''
        cityname = cityname.strip()
        cityname = cityname.replace(" ", "%20")
        cityname = cityname.lower()
        
        return cityname


    def tripadvisor_api_call(self, cityname: str):
        '''Calls the Tripadvisor API to search for the city and returns the JSON 
        response.
        
        Args:
            cityname (`str`): The name of the city to search for.
        
        Returns:
            response.json() (`dict`): The JSON response from the API call.
        '''
        cityname = self.prep_cityname_api(cityname)
        query_params = f"&searchQuery={cityname}" + "&category=geos&language=en"
        
        url = API_TRIP_LOC_SEARCH + f"key={TRIPADVISOR_API_KEY}" + query_params
        
        headers = {"accept": "application/json"}
        tripadvisor_response = requests.get(url, headers=headers)
        if tripadvisor_response.status_code == 403:
            if not self._api_key:
                raise ValueError("API key is missing.")
            else:
                raise HTTPError(f"{tripadvisor_response.status_code}: The Tripadvisor API key is likely missing this device's IP address. Head to https://www.tripadvisor.com/developers?screen=credentials to add this device's IP address to the API key.")
        if tripadvisor_response.status_code == 429:
            raise HTTPError(f"{tripadvisor_response.status_code}: The Tripadvisor API has reached its limit of requests. Please try again later.")
        
        return tripadvisor_response.json()


    def tripadvisor_city_info(self, cityname: str):
        """Extracts the location_id, name, secondary_name, and state from the
        Tripadvisor API response JSON for the provided city.
        
        Args:
            cityname (`str`): The name of the city to search for.
        
        Returns:
            city_info (`dict`): The city information from the Tripadvisor API response.
        """
        tripadvisor_response_json = self.tripadvisor_api_call(cityname)
        
        data = tripadvisor_response_json['data']
        
        if len(data) == 0:
            raise ValueError(f"Tripadvisor API returned no results for {cityname}")
        
        for index, city in enumerate(data):
            if city["name"].lower().strip() == cityname.lower().strip():
                city_info = city
                break
            
            if index == len(data) - 1:
                raise ValueError(f"{cityname} was not found via the Tripadvisor API")
        
        return city_info


    def city_info_values(self, city_info: str):
        """Extracts the location_id, name, secondary_name, and state from the
        Tripadvisor API response JSON for the provided city.
        
        Args:
            city_info (`str`): The city information from the Tripadvisor API response.
        
        Returns:
            location_id (`int`): The location id of the city.
            name (`str`): The name of the city.
            secondary_name (`str`): The secondary name of the city.
            state (`str`): The state of the city.
        """
        location_id = city_info["location_id"]
        name = city_info["name"].replace(" ", "_")
        
        if "city" in city_info["address_obj"]:
            secondary_name = city_info["address_obj"]["city"].replace(" ", "_")
        else:
            secondary_name = None
        state = city_info["address_obj"]["state"].replace(" ", "_")
        
        return location_id, name, secondary_name, state
        

    def tripadvisor_city_hotels(self, base_url: str = TRIP_HOTELLST_URL):
        """Creates the URL for the hotel list for the city.
        
        Args:
            base_url (`str`): The base URL for the hotel list.
        
        Returns:
            city_hotellst_url (`str`): The URL for the city's list of hotels.
        """
        location_id, name, secondary_name, state = self.city_info_values(
                self.city_info
        )
        if secondary_name:
            city_hotellst_url = base_url + f"{location_id}-{name}_{secondary_name}_{state}-Hotels.html"
        else:
            city_hotellst_url = base_url + f"{location_id}-{name}_{state}-Hotels.html"
        
        return city_hotellst_url


    def get_additional_hotelpage(self, city_hotellst_url: str):
        """Goes to the next page of the hotel list for the city.
        
        Args:
            city_hotellst_url (`str`): The URL of the current page of the city's    
                list of hotels.
        
        Returns:
            city_hotellst_url (`str`): The URL of the next page of the city's list
                of hotels.
        """
        g_location_id = city_hotellst_url.split("-")[1]

        if "_" in city_hotellst_url.split("-")[2]:
            city_hotellst_url = city_hotellst_url.replace(g_location_id, 
                    g_location_id + "-oa30"
        ) 
            
        else:
            oa_old = city_hotellst_url.split("-")[2]
            oa_old_num = int(oa_old[2:])
            glocationid_oa = g_location_id + "-" + oa_old
            oa_new_num = oa_old_num + 30
            oa_new = "oa" + str(oa_new_num)
            city_hotellst_url = city_hotellst_url.replace(glocationid_oa, 
                    g_location_id + "-" + oa_new
            )
        
        return city_hotellst_url


    def selen_crawl(self, url, save_path: Optional[str] = None, web_driver_install: bool = False):
        """Crawls a webpage using Selenium and returns the page source code.
        
        Args:
            url (`str`): The url of the webpage to scrape.
            save_path (`str`): The pathway to save the page source code.
            web_driver_install (`bool`): Whether to install the ChromeDriverManager.
        
        Returns:
            page_source_code (`BeautifulSoup`): The page source code of the webpage.
        """
        if web_driver_install:
            chrome_driver = webdriver.Chrome(ChromeDriverManager().install())
        else:
            chrome_driver = webdriver.Chrome()
        chrome_driver.get(url)
        chrome_driver.implicitly_wait(random.randint(60, 100))
        page_source_code = soup(chrome_driver.page_source, 'lxml')
        chrome_driver.quit()
        
        if save_path:
            save_path = save_path + ".pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(page_source_code, f)
                
        return page_source_code
                
                
    def smartproxy_residential_selen_crawl(self, url: str, save_path: Optional[str] = None):
        """Crawls a webpage using Selenium and Smartproxy residential proxies and
        returns the page source code.
        
        Args:
            url (`str`): The URL of the webpage to scrape.
            save_path (`str`): The pathway to save the page source code.
        
        Returns:
            page_source_code (`object`): The page source code of the webpage.
        """
        chrome_options = webdriver.ChromeOptions()

        proxies_extension = proxies(SP_RESI_USER, SP_RESI_PAS, 
                PROXY_HOST, PROXY_PORT
        )
        chrome_options.add_extension(proxies_extension)

        chrome_driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        chrome_driver.get(url)

        chrome_driver.implicitly_wait(random.randint(100, 150))

        page_source_code = soup(chrome_driver.page_source, 'lxml')
        chrome_driver.quit()
        
        if save_path:
            save_path = save_path + ".pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(page_source_code, f)
        
        return page_source_code


    def smartproxy_web_crawl(self, url: str, save_path: Optional[str] = None):
        """Uses Smartproxy Web Scrape to scrape a webpage and returns the page source 
        code.
        
        Args:
            url (`str`): The URL of the webpage to scrape.
            save_path (`str`): The pathway to save the page source code.
        
        Returns:
            page_source_code (`object`): The page source code of the webpage.
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
        page_source_code  = soup(html_string, 'lxml')
        if save_path:
            save_path = save_path + ".pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(page_source_code, f)
        
        return page_source_code


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
                raise HTTPError(f"{response.status_code}: The Tripadvisor API key is likely missing this device's IP address. Head to https://www.tripadvisor.com/developers?screen=credentials to add this device's IP address to the API key.")
        if response.status_code == 429:
            raise HTTPError(f"{response.status_code}: The Tripadvisor API has reached its limit of requests. Please try again later.")
        
        return response.json() 
    
    
    def get_hotel_info(self, page_source_code: object, hotel_lst: List[Dict]):
        """Creates a lists within a list of hotel information from the page source 
        code from a Tripadvisor city's list of hotels webpage.
        
        Args:
            page_source_code (`object`): The page source code from the city's list
                of hotels webpage.
            hotel_lst (`List[Dict]`): A list of dictionaries containing the hotel
                information.
        
        Returns:
            hotel_lst (`List[Dict]`): A list of dictionaries containing the hotel
                information.
        """
        hotel_cards = page_source_code.find_all('div', 
                {'data-automation': 'hotel-card-title'}
        )
        
        for hotel_card in hotel_cards:
            hotel_page_url = hotel_card.find('a')['href']
            location_id = hotel_page_url.split("-")[2].strip("d")
            hotel_json = self.hotel_api_call(int(location_id))
            hotel_dict = {k: hotel_json.get(k, None) for k in 
                    TRIP_LOC_RESPONSE_KEYS
            }
            hotel_dict["address_string"] = hotel_json["address_obj"]["address_string"]
            hotel_dict["city"] = hotel_json["address_obj"]["city"]
            if "street1" in hotel_json["address_obj"]:
                hotel_dict["street1"] = hotel_json["address_obj"]["street1"]
            else:
                hotel_dict["street1"] = np.nan
            hotel_lst.append(hotel_dict)
            
        pagination_text = page_source_code.find('div', class_='Ci').text
        numbers = list(map(int, re.findall(r'\d+', pagination_text)))
        interval_end_boundary = numbers[-2]
        total_city_results = numbers[-1]
        if interval_end_boundary < total_city_results:
            continue_crawl = True
        else:
            continue_crawl = False
        
        return hotel_lst, continue_crawl


    def crawl_hotels_lst(self, loc_hotellst_url: Optional[str] = None, 
                save_path: Optional[str] = False
        ):
        """Crawls the a city's list of hotels on Tripadvisor.
        
        Args:
            loc_hotellst_url (`str`): A URL which has a page of the city's list 
                of hotels.
            save_path (`str`): The pathway to save the page source code.
        
        Returns:
            hotel_lst (`List[Dict]`): A list of dictionaries containing the 
                hotel information.
        """
        if not loc_hotellst_url:
            loc_hotellst_url = self.initial_hotellst_url
        
        continue_crawl = True
        hotel_lst = []
        
        while continue_crawl:
            page_source_code = self.smartproxy_web_crawl(loc_hotellst_url, save_path)
            hotel_lst, continue_crawl = self.get_hotel_info(
                    page_source_code, hotel_lst
            )
            if continue_crawl:
                loc_hotellst_url = self.get_additional_hotelpage(loc_hotellst_url)
            else:
                continue_crawl = False
        
        return hotel_lst


    def get_room_number(self, hotel_lst: List[Dict] = None):
        """Use Smartproxy API to scrape the the hotel address and number of rooms 
        from the hotel's tripadvisor page.
        
        Args:
            hotel_lst (`List[Dict]`): The list of hotels to scrape the number
        
        Returns:
            hotel_lst (`List[Dict]`): The list of hotel dictionaries with 
                the number of rooms.

        """
        if not hotel_lst:
            hotel_lst = self.noroom_hotel_lst
        
        for index, hotel_dict in enumerate(hotel_lst):
            hotel_url = hotel_dict["web_url"]
            hotel_source_html = self.smartproxy_web_crawl(hotel_url, 
                    save_path = False
            )
            try:
                label_div = hotel_source_html.find('div', text='NUMBER OF ROOMS')
                number_of_rooms = label_div.find_next('div').text.strip() if label_div else np.nan
            except Exception as e:
                number_of_rooms = np.nan
                self._logger.error(f"Error: {e}.  Hotel_url: {hotel_url}")
            hotel_dict["number_of_rooms"] = number_of_rooms
            
            hotel_lst[index] = hotel_dict
        
        return hotel_lst


class CityGeo:
    def __init__(self, cityname: str, 
                boundaries_data_path: Path = BOUNDARIES_DATA_PATH
        ):
        self.cityname = cityname
        self.cityname_geojson = self.prep_cityname_geojson(cityname)
        self.city_geojson_pathway = boundaries_data_path / f"{self.cityname_geojson}.geojson"
        self.city_geojson = self.load_boundary_geojson()
        self.geojson_type = self.city_geojson['features'][0]['geometry']['type'].lower().strip()
        self.geojson_coordinates = self.city_geojson['features'][0]['geometry']['coordinates']
        self.geo = self.city_geo()
    
    
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

if __name__ == "__main__":
    Galveston = TripadvisorCityCrawl("Galveston")