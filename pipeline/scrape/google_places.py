"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union

# Third-party imports
from shapely import MultiPolygon, Polygon

# Application imports
from pipeline.scrape.common import IPlacesProvider


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

    def find_places_in_geography(self, geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
        """Locates all POIs within the given geography.

        Documentation: # TODO: Cite whatever resources you use here:
        - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """
        pass
