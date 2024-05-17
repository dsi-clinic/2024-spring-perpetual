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
import haversine as hs
import requests
from shapely import MultiPolygon, Polygon

# Application imports
from foodware.places.common import IPlacesProvider, Place, PlacesSearchResult
from common.geometry import BoundingBox, convert_meters_to_degrees


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


class GooglePlacesBusinessStatus(Enum):
    """Enumerates potential statuses for a business."""

    CLOSED_PERMANENTLY = "CLOSED_PERMANENTLY"
    CLOSED_TEMPORARILY = "CLOSED_TEMPORARILY"
    OPERATIONAL = "OPERATIONAL"


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

    def map_place(self, place: Dict) -> Place:
        """Maps a place fetched from a data source to a standard representation.

        Args:
            place (`dict`): The place.

        Returns:
            (`Place`): The standardized place.
        """
        id = place["id"]
        name = place["displayName"]["text"]
        categories = "|".join(place["types"])
        aliases = "|".join(place["types"])
        lat, lon = place["location"].values()
        address = place["formattedAddress"]
        is_closed = place.get("businessStatus") and (
            place["businessStatus"] != GooglePlacesBusinessStatus.OPERATIONAL.value
        )
        source = "google"

        return Place(
            id, name, categories, aliases, lat, lon, address, is_closed, source
        )

    def find_places_in_bounding_box(
        self,
        original_geom,
        box: BoundingBox,
        categories: List[str],
        search_radius: float,
    ) -> Tuple[List[Dict], List[Dict]]:
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
            ((`list` of `dict`, `list` of `dict`,)): A two-item tuple
                consisting of the list of retrieved places and a list
                of any errors that occurred, respectively.
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
        def send_request():
            return requests.post(url, params=params, headers=headers, json=body)

        def retry(func, max_retries: int = 3):
            retries = 0
            last_error = ValueError("Max retries must be >= 0.")
            while retries <= max_retries:
                try:
                    return func()
                except Exception as e:
                    self._logger.error(f"Request failed. {e}")
                    retries += 1
                    last_error = e
                    time.sleep(3)
            raise last_error

        r = retry(send_request)

        # Sleep and then parse JSON from response body
        try:
            time.sleep(GooglePlacesClient.SECONDS_DELAY_PER_REQUEST)
            data = r.json()
        except Exception as e:
            self._logger.error(f"Failed to parse reponse body JSON. {e}")
            return [], [{"body": body, "error": str(e)}]

        # If error occurred, store information and exit processing for cell
        if not r.ok or "error" in data:
            self._logger.error(
                "Failed to retrieve POI data through the Google Places API. "
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
                if not sub.intersects_with(original_geom):
                    continue
                sub_pois, sub_errs = self.find_places_in_bounding_box(
                    original_geom, sub, categories, search_radius / 2
                )
                pois.extend(sub_pois)
                errors.extend(sub_errs)
            return pois, errors

        # Otherwise, extract business data from response body JSON
        return data["places"], []

    def run_nearby_search(
        self, geo: Union[Polygon, MultiPolygon], categories: List[str]
    ) -> Tuple[List[Dict], List[Dict]]:
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
            (`PlacesResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)
        self.geo = geo

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

        # Determine initial search radius for cells
        top_left_cell = cells[0]
        center_pt = tuple(top_left_cell.center.to_list(coerce_to_float=True))
        bottom_right_pt = tuple(
            top_left_cell.bottom_right.to_list(coerce_to_float=True)
        )
        search_radius = hs.haversine(center_pt, bottom_right_pt, hs.Unit.METERS)

        # Batch categories to filter POIs in request
        batch_size = GooglePlacesClient.MAX_NUM_CATEGORIES_PER_REQUEST
        category_batches = (
            categories[i : i + batch_size]
            for i in range(0, len(categories), batch_size)
        )

        # Locate POIs within each cell if it contains any part of geography
        pois = []
        errors = []
        for batch in category_batches:
            for cell in cells:
                if cell.intersects_with(geo):
                    cell_pois, cell_errors = self.find_places_in_bounding_box(
                        original_geom=geo,
                        box=cell,
                        categories=batch,
                        search_radius=search_radius,
                    )
                    pois.extend(cell_pois)
                    errors.extend(cell_errors)

        cleaned_pois = self.clean_places(pois, geo)

        return PlacesSearchResult(pois, cleaned_pois, errors)

    def run_text_search(
        self, query: str, restriction: Polygon | MultiPolygon | None = None
    ) -> PlacesSearchResult:
        return super().run_text_search(query, restriction)
