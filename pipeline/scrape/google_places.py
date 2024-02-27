"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
import time
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Tuple, Union

# Third-party imports
import requests
# Application imports
from pipeline.scrape.common import IPlacesProvider
from pipeline.utils.geometry import (
    BoundingBox,
    convert_meters_to_degrees,
)


class GooglePOITypes(Enum):
    """Enumerates all relevant categories for points of interest."""

    # Potential Indoor Points
    BAR = "bar"
    RESTAURANT = "restaurant"

    # Potential Outdoor Points
    AIRPORT = "airport"
    BUS_STATION = "bus_station"
    COFFEE_SHOP = "cafe"
    COMMUNITY_CENTER = "community_center"
    CONVENIENCE_STORE = "convenience_store"
    DRUGSTORE = "drugstore"
    LIBRARY = "library"
    HOSPITAL = "hospital"
    HOTEL = "hotel"
    LIGHT_RAIL_STATION = "light_rail_station"
    LODGING = "lodging"
    MALL = "shopping_mall"
    MARKET = "market"
    MUSEUM = "museum"
    PARK = "park"
    PARKING = "parking"
    PHARMACY = "pharmacy"
    POST_OFFICE = "post_office"
    SCHOOL = "school"
    SUBWAY_STATION = "subway_station"
    SUPERMARKET = "supermarket"
    TRAIN_STATION = "train_station"
    TRANSIT_STATION = "transit_station"
    UNIVERSITY = "university"
    ZOO = "zoo"


class GooglePlacesBasicSKUFields(Enum):
    """Enumerates the place fields available under the Basic SKU."""

    ADDRESS_COMPONENTS = "places.addressComponents"
    BUSINESS_STATUS = "places.businessStatus"
    DISPLAY_NAME = "places.displayName"
    FORMATTED_ADDRESS = "places.formattedAddress"
    ID = "places.id"
    LOCATION = "places.location"
    PRIMARY_TYPE = "places.primaryType"
    PRIMARY_TYPE_DISPLAY_NAME = "places.primaryTypeDisplayName"
    SHORT_FORMATTED_ADDRESS = "places.shortFormattedAddress"
    SUB_DESTINATIONS = "places.subDestinations"
    TYPES = "places.types"


class GooglePlacesClient(IPlacesProvider):
    """A simple wrapper for the Google Places API (New)."""

    MAX_NUM_CATEGORIES_PER_REQUEST: int = 50
    """The maximum number of category filters permitted per request.
    """

    MAX_NUM_RESULTS_PER_REQUEST: int = 20
    """The maximum number of records that can be returned in a single query.
    """

    MAX_SEARCH_RADIUS_IN_METERS: float = 50_000
    """The maximum size of the search radius in meters. Approximately equal to 31 miles.
    """

    SECONDS_DELAY_PER_REQUEST: float = 0.5
    """The number of seconds to wait after each HTTP request.
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
                f'Missing expected environment variable "{e}".'
            ) from None

    def find_places_in_bounding_box(
        self, box: BoundingBox, categories: List[str], search_radius: float
    ) -> Tuple[Dict, Dict]:
        """Locates all POIs within the given area and categories.
        The area is further divided into a grid of quadrants if
        more results are available within the area than can be
        returned due to API limits.

        Args:
            box (`BoundingBox`): The bounding box.

            categories (`list` of `str`): The categories to search by.

            search_radius (`float`): The search radius, converted from
                meters to the larger of degrees longitude and latitude.

        Returns:
            (`dict`, `dict`): A two-item tuple consisting of the POIs and errors.
        """
        # Initialize request URL
        url = "https://places.googleapis.com/v1/places:searchNearby"

        # Build request params, body, and headers
        body = {
            "includedTypes": categories,
            "maxResultCount": GooglePlacesClient.MAX_NUM_RESULTS_PER_REQUEST,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": float(box.center.lat),
                        "longitude": float(box.center.lon),
                    },
                    "radius": search_radius,
                }
            },
        }
        params = {"key": self._api_key}
        headers = {
            "X-Goog-FieldMask": ",".join(
                str(e.value) for e in GooglePlacesBasicSKUFields
            ),
        }

        # Send POST request to the Google Places API
        r = requests.post(url, params=params, headers=headers, json=body)

        # Sleep and then parse JSON from response body
        time.sleep(GooglePlacesClient.SECONDS_DELAY_PER_REQUEST)
        data = r.json()

        # If error occurred, store information and exit processing for cell
        if not r.ok or "error" in data:
            self._logger.error(
                "Failed to retrieve POI data through the Yelp API. "
                f'Received a "{r.status_code}-{r.reason}" status code '
                f'with the message "{r.text}".'
            )
            return [], [{"body": body, "error": data}]

        # Otherwise, if no data returned, return empty lists of POIs and errors
        if not data:
            self._logger.warning("No data found in response body.")
            return [], []

        # Otherwise, if number of POIs returned equals max,
        # split box and recursively issue HTTP requests
        if len(data["places"]) == GooglePlacesClient.MAX_NUM_RESULTS_PER_REQUEST:
            pois = []
            errors = []
            sub_cells = box.split_along_axes(x_into=2, y_into=2)
            for sub in sub_cells:
                sub_pois, sub_errs = self.find_places_in_bounding_box(
                    sub, categories, search_radius / 2
                )
                pois.extend(sub_pois)
                errors.extend(sub_errs)
            return pois, errors

        # Otherwise, extract business data from response body JSON
        return data["places"], []

    def find_places_in_geography(self, geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
        """Locates all POIs with a review within the given geography.
        The Google Places API permits searching for POIs within a radius around
        a given point. Therefore, data is extracted by dividing the
        geography's bounding box into cells of equal size and then searching
        within the circular areas that circumscribe (i.e., perfectly enclose)
        those cells.

        To circumscribe a cell, the circle must have a radius that is
        one-half the length of the cell's diagonal (as derived from the
        Pythagorean Theorem). Let `side` equal the length of a cell's side.
        It follows that the radius is:

        ```
        radius = (√2/2) * side
        ```

        Google Places sets a cap on the radius search size, so after solving for `side`,
        it follows that cell sizes are restricted as follows:

        ```
        max_side = √2 * max_radius
        ```

        Therefore, the bounding box must be split into _at least_ the following
        number of cells along the x- and y- (i.e., longitude and latitude)
        directions to avoid having cells that are too big:

        ```
        min_num_splits = ceil(bounding_box_length / max_side)
        ```

        Finally, at the time of writing, only 20 records are returned per
        search query, even if more businesses are available. Therefore, it
        is important to confirm that less than 20 records are returned
        in the response to avoid missing data.

        Documentation:
            - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)
            - ["Nearby Search"](https://developers.google.com/maps/documentation/places/web-service/search-nearby)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`list` of `dict`): The list of places.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)

        # Calculate length of square circumscribed by circle with the max search radius
        max_side_meters = (2**0.5) * GooglePlacesClient.MAX_SEARCH_RADIUS_IN_METERS

        # Use heuristic to convert length from meters to degrees at box's lower latitude
        deg_lat, deg_lon = convert_meters_to_degrees(max_side_meters, bbox.bottom_left)

        # Take minimum value as side length (meters convert differently to
        # lat and lon, and we want to avoid going over max radius)
        max_side_degrees = min(deg_lat, deg_lon)

        # Divide box into grid of cells of approximately equal length and width
        # NOTE: Small size differences may exist due to rounding.
        cells: List[BoundingBox] = bbox.split_into_squares(
            size_in_degrees=Decimal(str(max_side_degrees))
        )

        # Batch categories to filter POIs in request
        categories = [str(e.value) for e in GooglePOITypes]
        batch_size = GooglePlacesClient.MAX_NUM_CATEGORIES_PER_REQUEST
        category_batches = (
            categories[i : i + batch_size]
            for i in range(0, len(GooglePOITypes), batch_size)
        )

        # Locate POIs within each cell if it contains any part of geography
        pois = []
        errors = []
        for batch in category_batches:
            for cell in cells:
                if cell.intersects_with(geo):
                    cell_pois, cell_errors = self.find_places_in_bounding_box(
                        box=cell,
                        categories=batch,
                        search_radius=GooglePlacesClient.MAX_SEARCH_RADIUS_IN_METERS,
                    )
                    pois.extend(cell_pois)
                    errors.extend(cell_errors)

        return pois, errors
