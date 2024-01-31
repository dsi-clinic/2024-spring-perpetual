"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union

# Third-party imports
import requests
import time
from shapely.geometry import MultiPolygon, Polygon, Point
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
        
        
    def find_places_in_geography(self, geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
        """Locates all POIs within the given geography."""
        # Get bounding box from the input geometry
        min_lon, min_lat, max_lon, max_lat = self.get_bounding_box_from_geometry(geo)

        # Generate quadrants 
        n_lon, n_lat = 9, 9  # Creates 9x9 quadrants, readjust 
        _, quadrants = generate_quadrants(min_lon, min_lat, max_lon, max_lat, n_lon, n_lat)

        # Calculate center points for each quadrant
        center_points = calculate_center_points(quadrants)
     
        # Initialize API URL and parameters
        api_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        all_pois = []

        # For each center point, interact with the Google Places API
        for center_point in center_points:
            params = {
            'location': f'{center_point.y},{center_point.x}',  # latitude,longitude
            'radius': 1000,  # in meters
            'key': self._api_key
        }

        while True:
            response = requests.get(api_url, params=params)
            if response.status_code != 200:
                self._logger.error(f"Error from Google Places API: {response.status_code} - {response.reason}")
                break  # Optionally, could raise an exception here

            data = response.json()
            all_pois.extend(data.get('results', []))

            # Check for the 'next_page_token' for pagination
            page_token = data.get('next_page_token')
            if not page_token:
                break  # No more pages
            params['pagetoken'] = page_token

            # Google requires a short delay before fetching the next page
            time.sleep(2)

        return all_pois
       

    """Documentation: # TODO: Cite whatever resources you use here:
        - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """