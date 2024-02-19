"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
import time
from typing import Dict, List, Union

# Third-party imports
import requests
# Application imports
from pipeline.scrape.common import IPlacesProvider
from pipeline.utils.geometry import gplaces_get_geojson_centerpoints

from shapely.geometry import MultiPolygon, Polygon


class GooglePlacesClient(IPlacesProvider):
    """A simple wrapper for the Google Places API."""

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
                f'Missing expected environment variable "{e}".'
            ) from None

    def find_places_in_geography(
        self, geo: Union[Polygon, MultiPolygon], place_type: str
        ) -> List[Dict]:
        """Locates all POIs within the given geography.

        Uses the Google Places API to find places of a specific 
        type within a geographic area. The area is divided into a 
        grid of quadrants to manage the scope of each API call.

        Documentation: # TODO: Cite whatever resources you use here:
            - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)
            - ["Nearby Search"](https://developers.google.com/maps/documentation/places/web-service/search-nearby)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """

        # Initialize API base URL and parameters
        api_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

        # Input for getting centerpoints
        filepath = 'data/boundaries/hilo.geojson'

        all_pois = []
        
        # Calculate center points
        center_points = gplaces_get_geojson_centerpoints(filepath,2,2)
        
        # Iterate over each center point
        for point in center_points:
            point_lat, point_long = point # Extract latitude and longitude
            
            # Iterate over each type in place_types list - removed for now
                # Prepare paramaters for API call
            params = {
                "location": f"{point_lat},{point_long}",  # Latitude,longitude
                "radius": 10000,  # In meters
                "type": place_type,  # Type of place you are looking for (category filtering)
                "key": self._api_key,
            }
                 
            # Make a request to the Google PLaces API with the current set of parameters
            response = requests.get(api_url, params=params)

            # Check that the scrape worked
            if response.status_code == 200:
                #print("Request successful.")
                # Parse the response data from JSON format
                data = response.json()
                #print(data)
                    
                # Google uses results to list places
                places = data.get("results", [])
                for place in places:
                    place_info = {
                        'name': place.get('name'),
                        'coordinates': place.get('geometry', {}).get('location', {}), 
                        'location': place.get('vicinity'),
                        'types': place.get('types'),
                    }
                    all_pois.append(place_info)
            else:
                # Log the error or handle it according to your needs
                    self._logger.error(f"Error from Google Places API: {response.status_code} - {response.reason}")
                    break  # Exit the loop on error

        # Return the list of all POIs found in all API requests
        return all_pois
