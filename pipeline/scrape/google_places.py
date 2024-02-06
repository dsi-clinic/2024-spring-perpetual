"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union
import geojson

# Third-party imports
import requests
import time
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd

# Application imports
from pipeline.scrape.common import IPlacesProvider
from pipeline.utils.geometry import generate_quadrants, calculate_center_points, get_bounding_box_from_geometry

class GooglePlacesClient(IPlacesProvider):
    """A simple wrapper for the Google Places API.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `GooglePlacesClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if an environment variable, 
                `GOOGLE_MAPS_API_KEY`, is not found.
        
        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["GOOGLE_MAPS_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GooglePlacesClient."
                f"Missing expected environment variable \"{e}\"."
            ) from None
        
        
    def find_places_in_geography(self, geo: Union[Polygon, MultiPolygon], place_type: str) -> List[Dict]:
        """Locates all POIs within the given geography.

        Uses the Google Places API to find places of a specific type within a geographic area. 
        The area is divided into a grid of quadrants to manage the scope of each API call.

        Documentation: # TODO: Cite whatever resources you use here:
            - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)
            - ["Nearby Search"](https://developers.google.com/maps/documentation/places/web-service/search-nearby)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """
        # Get the bounding box from the input geometry
        min_lon, min_lat, max_lon, max_lat = get_bounding_box_from_geometry(geo)

        # Generates quadrants to divide the bounding box into.
        n_lon, n_lat = 9, 9 # 9x9 grid
        _, quadrants = generate_quadrants(min_lon, min_lat, max_lon, max_lat, n_lon, n_lat)

        # Calculate center points for each quadrant
        center_points = calculate_center_points(quadrants)
     
        # Initialize API base URL and parameters
        api_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        # Initialize a list to store all found POIs
        all_pois = []

        # For each center point, interact with the Google Places API
        for center_point in center_points:
            # Initialize parameters for the new center point, effectively resetting the pagetoken
            params = {
            'location': f'{center_point.y},{center_point.x}',  # Latitude,longitude
            'radius': 1000,  # In meters
            'type': place_type, # Type of place you are looking for (category filtering)
            'key': self._api_key
            }

            # Initialize or reset pagination token
            page_token = None
            # Initialize a counter for the number of pages fetched
            pages_fetched = 0
            max_pages = 3 # Maximum number of pages to fetch

            # Initialize a counter for consecutive errors
            consecutive_errors = 0
            max_errors_allowed = 5 # Max number of consecutive errors allowed
        
            # Loop to handle pagination in the API response
            while True:
                if page_token:
                    # If there's a page token from a previous request, add it to the parameters
                    params['pagetoken'] = page_token
                else:
                    # Ensure the pagetoken paramter is removed if not needed
                    params.pop('pagetoken', None) # Remove the pagetoken from params if it exists

                # Make a request to the Google PLaces API with the current set of parameters
                response = requests.get(api_url, params=params)

                if response.status_code != 200:
                    # If the API response is not successful log the error and stop processing
                    self._logger.error(f"Error from Google Places API: {response.status_code} - {response.reason}")
                    consecutive_errors += 1
                    if consecutive_errors > max_errors_allowed:
                        break  # Optionally, could raise an exception here
                else: 
                    consecutive_errors = 0 # Reset the error counter on successful response     
                
                # Parse the response data from JSON format
                data = response.json()
                # Extend the all_pois list with the places found in the current request
                all_pois.extend(data.get('results', []))

                # Increment the pages_fetched counter and check against max_pages
                pages_fetched += 1
                if pages_fetched >= max_pages:
                    break # Break if maximum number of pages is reached

                # Check for the 'next_page_token' for pagination
                page_token = data.get('next_page_token')
                if not page_token:
                    # If there's no next_page_token, it means there are no more pages of results
                    # Break from the loop and proceed to the next center point
                    break

                # Google requires a short delay before fetching the next page
                time.sleep(2)

        # Return the list of all POIs found in all API requests
        return all_pois
    

if __name__ == "__main__":

    gplaces_scrape = GooglePlacesClient(IPlacesProvider)
    #path determined from config input 
    path = "../../data/boundaries/hilo.geojson"
    #place_type determined from config input 
    place_type = "Restaurants"

    with open(path) as f:
        gf = geojson.load(f)
    geo = gf['coordinates']
    
    gplaces_scrape.find_places_in_geography(geo, place_type)

