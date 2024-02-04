"""Provides access to geographic locations using the Microsoft Bing Maps API.
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


class BingMapsClient(IPlacesProvider):
    """A simple wrapper for the Microsoft Bing Maps API."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `BingMapsClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if an environment variable,
                `MICROSOFT_BING_API_KEY`, is not found.

        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["MICROSOFT_BING_API_KEY"]
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
        - ["Bing Maps | Local Search"](https://learn.microsoft.com/en-us/bingmaps/rest-services/locations/local-search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The places.
        """
        pass
