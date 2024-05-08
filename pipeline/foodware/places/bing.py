"""Provides access to geographic locations using the Microsoft Bing Maps API.
"""

# Standard library imports
import logging
import os
import time
from enum import Enum
from typing import Dict, List, Tuple, Union

# Third-party imports
import requests
from shapely import MultiPolygon, Polygon

# Application imports
from foodware.places.common import IPlacesProvider, Place, PlacesSearchResult
from common.geometry import BoundingBox


class BingPOICategories(Enum):
    """Enumerates all relevant categories for points of interest."""

    # Potential Indoor Points
    EAT_DRINK = "EatDrink"

    # Potential Outdoor Points
    HOSPITALS = "Hospitals"
    HOTELS_AND_MOTELS = "HotelsAndMotels"
    SEE_DO = "SeeDo"
    SHOPPING_CENTERS = "MallsAndShoppingCenters"


class BingMapsClient(IPlacesProvider):
    """A simple wrapper for the Microsoft Bing Maps API."""

    DEFAULT_SEARCH_GRID: Tuple[int, int] = (
        2,
        2,
    )
    """The default number of cells to generate in a bounding box used for POI search.
    """

    SECONDS_DELAY_PER_RATE_LIMIT: float = 10
    """The default number of seconds to wait after being rate limited.
    """

    SECONDS_DELAY_PER_REQUEST: float = 0.5
    """The default number of seconds to wait in between successive calls to the API.
    """

    MAX_NUM_QUERY_RESULTS: int = 25
    """The maximum number of results that can be returned from a single query.
    """

    RATE_LIMITED_FLAG: str = "X-MS-BM-WS-INFO"
    """A flag added to the header of an HTTP response to indicate that the
    request has been rate limited. For more information, please refer to the
    following [product documentation](https://learn.microsoft.com/en-us/bingmaps/getting-started/bing-maps-api-best-practices).
    """

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
                "Failed to initialize BingMapsClient."
                f'Missing expected environment variable "{e}".'
            ) from None

    def map_place(self, place: Dict) -> Place:
        """Maps a place fetched from a data source to a standard representation.
        NOTE: The Bing Maps API does not provide any ids for its places, so
        an id has been manually created by hashing a combination of fields
        presumed to be unique: the place name, latitude, and longitude.

        Args:
            place (`dict`): The place.

        Returns:
            (`Place`): The standardized place.
        """
        name = place["name"]
        categories = place["entityType"]
        aliases = place["entityType"]
        lat, lon = place["point"]["coordinates"]
        address = place["Address"]["formattedAddress"]
        is_closed = False
        source = "bing"
        id = str(hash(name + str(lat) + str(lon)))
        url = place.get("Website")

        return Place(
            id, name, categories, aliases, lat, lon, address, is_closed, source, url
        )

    def find_places_in_bounding_box(
        self, box: BoundingBox
    ) -> Tuple[List[Dict], List[Dict]]:
        """Locates all POIs within the bounding box.

        Args:
            box (`BoundingBox`): The bounding box.

            categories (`list` of `str`): The categories to search by.

        Returns:
            ((`list` of `dict`, `list` of `dict`,)): A two-item tuple
                consisting of the list of retrieved places and a list
                of any errors that occurred, respectively.
        """
        # Initialize request URL and static params
        url = "https://dev.virtualearth.net/REST/v1/LocalSearch/"
        categories = ",".join(e.value for e in BingPOICategories)
        limit = BingMapsClient.MAX_NUM_QUERY_RESULTS

        # Issue POI query for bounding box
        pois = []
        errors = []
        while True:
            # Build request parameters and headers
            sw_lat, sw_lon = [str(float(c)) for c in box.bottom_left.to_list()]
            ne_lat, ne_lon = [str(float(c)) for c in box.top_right.to_list()]
            view = ",".join([sw_lat, sw_lon, ne_lat, ne_lon])
            params = {
                "type": categories,
                "maxResults": limit,
                "userMapView": view,
                "key": self._api_key,
            }
            headers = {"Accept": "application/json"}

            # Send request, parse JSON response, and wait to abide by rate limit
            r = requests.get(url, headers=headers, params=params)
            data = r.json()
            time.sleep(BingMapsClient.SECONDS_DELAY_PER_REQUEST)

            # If error occurred, store information and exit processing for cell
            if not r.ok:
                err_msg = " ".join(data["errorDetails"])
                self._logger.error(
                    "Failed to retrieve POI data through the TomTom API. "
                    f'Received a "{r.status_code}-{r.reason}" status code '
                    f'with the message "{err_msg}".'
                )
                errors.append({"params": params, "error": err_msg})
                return pois, errors

            # If rate limited, wait a few seconds and reattempt call
            if int(r.headers[BingMapsClient.RATE_LIMITED_FLAG]):
                time.sleep(BingMapsClient.SECONDS_DELAY_PER_RATE_LIMIT)
                continue

            # If number of POIs returned is equal to max/limit,
            # split box and recursively issue HTTP requests
            if data["resourceSets"][0]["estimatedTotal"] == limit:
                sub_cells = box.split_along_axes(x_into=2, y_into=2)
                for sub in sub_cells:
                    sub_pois, sub_errs = self.find_places_in_bounding_box(sub)
                    pois.extend(sub_pois)
                    errors.extend(sub_errs)
                return pois, errors

            # Otherwise, extract business data from request body JSON
            return data["resourceSets"][0]["resources"], []

    def run_nearby_search(
        self, geo: Union[Polygon, MultiPolygon]
    ) -> PlacesSearchResult:
        """Queries the Bing Maps Local Search API for locations
        within a geographic boundary. To accomplish this, a bounding
        box for the geography is calculated and then split into
        many smaller boxes, each of which submitted to the API as
        a data query.

        Documentation:
        - ["Bing Maps | Local Search"](https://learn.microsoft.com/en-us/bingmaps/rest-services/locations/local-search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`PlacesResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)

        # Divide geography into grid of cells corresponding to separate POI queries
        num_x, num_y = BingMapsClient.DEFAULT_SEARCH_GRID
        cells = bbox.split_along_axes(x_into=num_x, y_into=num_y)

        # Locate POIs within each cell if it contains any part of geography
        pois = []
        errors = []
        for cell in cells:
            if cell.intersects_with(geo):
                cell_pois, cell_errors = self.find_places_in_bounding_box(cell)
                pois.extend(cell_pois)
                errors.extend(cell_errors)
            if pois:
                break

        # Clean POIs
        cleaned_pois = self.clean_places(pois, geo)

        return PlacesSearchResult(pois, cleaned_pois, errors)
