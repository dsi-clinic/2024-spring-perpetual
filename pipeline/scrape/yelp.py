"""Provides access to geographic locations using the Yelp Businesses API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union

# Third-party imports
import requests
# Application imports
from pipeline.scrape.common import IPlacesProvider
from shapely import MultiPolygon, Polygon
from pipeline.utils.geometry import get_geojson_centerpoints


class YelpClient(IPlacesProvider):
    """A simple wrapper for the Yelp Businesses API."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `YelpClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if an environment variable,
                `YELP_API_KEY`, is not found.

        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["YELP_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GooglePlacesClient."
                f'Missing expected environment variable "{e}".'
            ) from None

    def find_places_in_geography(
        self, geo: Union[Polygon, MultiPolygon]
    ) -> List[Dict]:
        """Locates all POIs within the given geography.

        Documentation: # TODO: Cite whatever resources you use here:
        - ["Yelp API Reference | Search"](https://docs.developer.yelp.com/reference/v3_business_search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """

        ''' api link '''
        url = "https://api.yelp.com/v3/businesses/search"
        
        '''categories to exlpore'''
        category_list = ['restaurant', 'bar', 'pharmacy', 'grocery']

        '''input for getting centerpoints'''
        # filepath = '../../data/boundaries/hilo.geojson'
        #fixing filepath bc the file is being run from a different directory
        # AKA access point is different, so filepath needed to be changed
        filepath = 'data/boundaries/hilo.geojson'

        '''running the function to get center points'''
        center_points  = get_geojson_centerpoints(filepath,2,2)
        
        ''' getting all businesses from center points'''
        POIs = []
        for point in center_points:
            point_lat, point_long = point[1], point[0]  # Extract latitude and longitude from the tuple
            params = {
                    'radius': 10000,
                    'categories': ','.join(category_list),
                    'longitude': point_long,
                    'latitude': point_lat}
            
            headers = {
                'Authorization': f'Bearer {self._api_key}',}
            response = requests.get(url, headers=headers, params=params)
            
            ''' making sure the scrape worked'''
            if response.status_code == 200:
                data = response.json()
                '''extracting information from json file'''
                businesses = data.get('businesses', [])
                for business in businesses:
                    place_info = {
                        'name': business.get('name'),
                        'coordinates': business.get('coordinates'),
                        'location': business.get('location'),}
                    POIs.append(place_info)
            else:
                '''error if request didn't work'''
                print(f"Failed to retrieve data. Status code: {response.status_code}")
        
        return POIs
