"""Provides access to geographic locations using the Yelp Businesses API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union

# Third-party imports
import requests
from shapely import MultiPolygon, Polygon

# Application imports
from pipeline.scrape.common import IPlacesProvider


class YelpClient(IPlacesProvider):
    """A simple wrapper for the Yelp Businesses API.
    """

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

    '''use api key to make call, make config file with key
    give any parameters that yelp might need, '''

        try:
            self._api_key = os.environ["YELP_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GooglePlacesClient."
                f"Missing expected environment variable \"{e}\"."
            ) from None
        
        
    def find_places_in_geography(
        self, 
        geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
        """Locates all POIs within the given geography.

        Documentation: # TODO: Cite whatever resources you use here:
        - ["Yelp API Reference | Search"](https://docs.developer.yelp.com/reference/v3_business_search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """
        pass
