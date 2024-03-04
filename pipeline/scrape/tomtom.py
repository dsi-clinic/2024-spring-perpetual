"""Provides access to geographic locations using the TomTom Search API.
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
from pipeline.scrape.common import IPlacesProvider
from pipeline.utils.geometry import BoundingBox


class TomTomPOICategories(Enum):
    """Enumerates all relevant categories for points of interest."""

    # Potential Indoor Points
    CAFE_PUB = 9376
    RESTAURANT = 7315
    NIGHTLIFE = 9379

    # Potential Outdoor Points
    DRUG_STORE = 9361051
    HIGH_SCHOOL = 7372006
    HOTEL = 7314003
    GROCERY_STORE = 9361023
    MIDDLE_SCHOOL = 7372014
    MOTEL = 7314006
    PHARMACY = 7326
    ELEMENTARY_OR_JUNIOR_HIGH_SCHOOL = 7372005
    PARK = 9362008
    PUBLIC_MARKET = 7332003
    PUBLIC_TRANSIT_STOP = 9942
    RESIDENTIAL_ACCOMMODATIONS = 7303
    RESORT = 7314005
    SENIOR_HIGH_SCHOOL = 7372007
    SHOPPING_CENTER = 7373
    SUPERMARKET_OR_HYPERMARKET = 7332005


class TomTomSearchClient(IPlacesProvider):
    """A simple wrapper for the TomTom Search API."""

    DEFAULT_SEARCH_GRID: Tuple[int, int] = (
        2,
        2,
    )
    """The default number of cells to generate in a bounding box used for POI search.
    """

    MAX_NUM_CATEGORIES_PER_REQUEST: int = 10
    """The maximum number of category filters permitted per request.
    """

    MAX_NUM_PAGE_RESULTS: int = 100
    """The maximum number of results that can be returned from a single HTTP request.
    """

    SECONDS_DELAY_PER_REQUEST: float = 0.2
    """The number of seconds to wait after each HTTP request.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `TomTomSearchClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if an environment variable,
                `TOMTOM_API_KEY`, is not found.

        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["TOMTOM_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize TomTomSearchClient."
                f'Missing expected environment variable "{e}".'
            ) from None

    def find_places_in_bounding_box(
        self, box: BoundingBox, categories: List[str]
    ) -> Tuple[Dict, Dict]:
        """Locates all POIs within the bounding box.

        Args:
            box (`BoundingBox`): The bounding box.

            categories (`list` of `str`): The categories to search by.

        Returns:
            (`dict`, `dict`): A two-item tuple consisting of the POIs and errors.
        """
        # Initialize request URL and static params
        url = "https://api.tomtom.com/search/2/poiSearch/.json"
        limit = TomTomSearchClient.MAX_NUM_PAGE_RESULTS

        # Issue POI query for bounding box
        pois = []
        errors = []
        page_idx = 0
        while True:
            # Build request parameters and headers
            params = {
                "key": self._api_key,
                "limit": limit,
                "ofs": page_idx * limit,
                "categorySet": categories,
                "topLeft": ",".join(
                    str(float(d)) for d in box.top_left.to_list(as_lat_lon=True)
                ),
                "btmRight": ",".join(
                    str(float(d)) for d in box.bottom_right.to_list(as_lat_lon=True)
                ),
            }
            headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}

            # Send request, parse JSON response, and wait to abide by rate limit
            r = requests.get(url, headers=headers, params=params)
            data = r.json()
            time.sleep(TomTomSearchClient.SECONDS_DELAY_PER_REQUEST)

            # If error occurred, store information and exit processing for cell
            if not r.ok:
                self._logger.error(
                    "Failed to retrieve POI data through the TomTom API. "
                    f'Received a "{r.status_code}-{r.reason}" status code '
                    f'with the message "{r.text}".'
                )
                errors.append({"params": params, "error": data})
                return pois, errors

            # Otherwise, extract business data from response body JSON
            page_pois = data.get("results", [])
            for poi in page_pois:
                pois.append(poi)

            # Determine total number of pages of data for query
            num_pages = (data["summary"]["totalResults"] // limit) + (
                1 if data["summary"]["totalResults"] % limit > 0 else 0
            )

            # Return POIs and errors if on last page
            if (not num_pages) or (page_idx == num_pages - 1):
                return pois, errors

            # Otherwise, iterate page index and add delay before next request
            page_idx += 1

    def find_places_in_geography(self, geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
        """Queries the TomTom Points of Interest Search API for
        locations within a geography boundary. To accomplish this,
        a bounding box for the geography is calculated and then
        split into many smaller boxes, each of which submitted to
        the API as a data query. At present, the number of boxes
        searched is equal to the maximum number of search requests
        that can be submitted simultaneously through the TomTom
        Asynchronous Batch API.

        Documentation:
        - ["Asynchronous Batch Submission | POST Body Fields | Query"](https://developer.tomtom.com/batch-search-api/documentation/asynchronous-batch-submission#post-body-fields)
        - ["Points of Interest Search"](https://developer.tomtom.com/search-api/documentation/search-service/points-of-interest-search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The places.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)

        # Divide geography into grid of cells corresponding to separate POI queries
        num_x, num_y = TomTomSearchClient.DEFAULT_SEARCH_GRID
        cells = bbox.split_along_axes(x_into=num_x, y_into=num_y)

        # Batch categories to filter POIs in request
        categories = [str(e.value) for e in TomTomPOICategories]
        batch_size = TomTomSearchClient.MAX_NUM_CATEGORIES_PER_REQUEST
        category_batches = (
            categories[i : i + batch_size]
            for i in range(0, len(TomTomPOICategories), batch_size)
        )

        # Locate POIs within each cell if it contains any part of geography
        pois = []
        errors = []
        for batch in category_batches:
            for cell in cells:
                if cell.intersects_with(geo):
                    cell_pois, cell_errors = self.find_places_in_bounding_box(
                        box=cell, categories=",".join(batch)
                    )
                    pois.extend(cell_pois)
                    errors.extend(cell_errors)

        return pois, errors
