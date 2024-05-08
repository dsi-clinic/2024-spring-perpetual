"""Provides access to geographic locations using the Yelp Fusion API.
"""

# Standard library imports
import logging
import math
import os
import time
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Tuple, Union

# Third-party imports
import requests
from shapely import MultiPolygon, Polygon

# Application imports
from foodware.places.common import IPlacesProvider, Place, PlacesSearchResult
from common.geometry import BoundingBox, convert_meters_to_degrees


class YelpPOICategories(Enum):
    """Enumerates all relevant categories for points of interest."""

    # Potential Indoor Points
    BAR = "bars"
    RESTAURANT = "restaurants"

    # Potential Outdoor Points
    ART = "arts"
    AIRPORT = "airports"
    APARTMENT = "apartments"
    BIKE_SHARING_HUB = "bikesharing"
    BUS_STATION = "busstations"
    CIVIC_CENTER = "civiccenter"
    COMMUNITY_CENTER = "communitycenters"
    COLLEGE_OR_UNIVERSITY = "collegeuniv"
    CONDO = "condominiums"
    DRUGSTORE = "drugstores"
    ELEMENTARY_SCHOOL = "elementaryschools"
    GROCERY = "grocery"
    HOSPITAL = "hospitals"
    HOTEL = "hotels"
    JUNIOR_OR_SENIOR_HIGH_SCHOOL = "highschools"
    LIBRARY = "libraries"
    MEDICAL_CENTER = "medcenters"
    METRO_STATION = "metrostations"
    OFFICE = "sharedofficespaces"
    PARK = "parks"
    POST_OFFICE = "postoffices"
    PHARMACY = "pharmacy"
    PRESCHOOL = "preschools"
    RECYCLING_CENTER = "recyclingcenter"
    RESORT = "resorts"
    SHARED_LIVING = "housingcooperatives"
    TRAIN_STATION = "trainstations"
    ZOO = "zoos"


class YelpClient(IPlacesProvider):
    """A simple wrapper for the Yelp Fusion API."""

    MAX_NUM_PAGE_RESULTS: int = 50
    """The maximum number of results that can be returned on a single page of
    search results. The inclusive upper bound of the "limit" query parameter.
    """

    MAX_NUM_QUERY_RESULTS: int = 1_000
    """The maximum number of results that can be returned from a single query.
    """

    MAX_SEARCH_RADIUS_IN_METERS: int = 40_000
    """The maximum size of the suggested search radius in meters.
    Approximately equal to 25 miles.
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
        try:
            self._api_key = os.environ["YELP_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize YelpClient."
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
        name = place["name"]
        categories = "|".join(c["title"] for c in place["categories"])
        aliases = "|".join(c["alias"] for c in place["categories"])
        lat, lon = place["coordinates"].values()
        address = ", ".join(place["location"]["display_address"])
        is_closed = place["is_closed"]
        source = "yelp"
        url = place["url"]

        return Place(
            id, name, categories, aliases, lat, lon, address, is_closed, source, url
        )

    def find_places_in_bounding_box(
        self, box: BoundingBox, search_radius: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """Locates all POIs within the bounding box.

        Args:
            box (`BoundingBox`): The bounding box.

            search_radius (`int`): The search radius, converted from
                meters to the larger of degrees longitude and latitude
                and rounded up to the nearest whole number.

        Returns:
            ((`list` of `dict`, `list` of `dict`,)): A two-item tuple
                consisting of the list of retrieved places and a list
                of any errors that occurred, respectively.
        """
        # Initialize request URL and static params
        url = "https://api.yelp.com/v3/businesses/search"
        categories = ",".join(e.value for e in YelpPOICategories)
        limit = YelpClient.MAX_NUM_PAGE_RESULTS

        # Issue POI query for minimum bounding circle circumscribing each box
        # NOTE: Only integers are accepted for the radius.
        pois = []
        errors = []
        page_idx = 0
        while True:
            # Build request parameters and headers
            params = {
                "radius": math.ceil(search_radius),
                "categories": categories,
                "longitude": float(box.center.lon),
                "latitude": float(box.center.lat),
                "limit": limit,
                "offset": page_idx * limit,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }

            # Send request and parse JSON response
            r = requests.get(url, headers=headers, params=params)
            data = r.json()

            # If error occurred, store information and exit processing for cell
            if not r.ok:
                self._logger.error(
                    "Failed to retrieve POI data through the Yelp API. "
                    f'Received a "{r.status_code}-{r.reason}" status code '
                    f'with the message "{r.text}".'
                )
                errors.append({"params": params, "error": data})
                return pois, errors

            # Otherwise, if number of POIs returned exceeds max, split
            # box and recursively issue HTTP requests
            if data["total"] > YelpClient.MAX_NUM_QUERY_RESULTS:
                sub_cells = box.split_along_axes(x_into=2, y_into=2)
                for sub in sub_cells:
                    sub_pois, sub_errs = self.find_places_in_bounding_box(
                        sub, search_radius / 4
                    )
                    pois.extend(sub_pois)
                    errors.extend(sub_errs)
                return pois, errors

            # Otherwise, extract business data from response body JSON
            page_pois = data.get("businesses", [])
            for poi in page_pois:
                pois.append(poi)

            # Determine total number of pages of data for query
            num_pages = (data["total"] // limit) + (
                1 if data["total"] % limit > 0 else 0
            )

            # Return POIs and errors if on last page
            if page_idx == num_pages - 1:
                return pois, errors

            # Otherwise, iterate page index and add delay before next request
            page_idx += 1
            time.sleep(0.5)

    def run_nearby_search(
        self, geo: Union[Polygon, MultiPolygon]
    ) -> PlacesSearchResult:
        """Locates all POIs with a review within the given geography.
        The Fusion API permits searching for POIs within a radius around
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

        Yelp sets a cap on the radius search size, so after solving for `side`,
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

        Finally, only 1,000 records may be fetched from a single query, with
        a maximum limit of 50 records per page of data, even if more businesses
        are available. Therefore, it is important to confirm that less than
        1,000 records are returned with a query to avoid missing data.

        Documentation:
        - ["Yelp API Reference | Search"](https://docs.developer.yelp.com/reference/v3_business_search)

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`PlacesResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)

        # Calculate length of square circumscribed by circle with the max search radius
        max_side_meters = (2**0.5) * YelpClient.MAX_SEARCH_RADIUS_IN_METERS

        # Use heuristic to convert length from meters to degrees at box's lower latitude
        deg_lat, deg_lon = convert_meters_to_degrees(max_side_meters, bbox.bottom_left)

        # Take minimum value as side length (meters convert differently to lat and lon,
        # and we want to avoid going over max radius)
        max_side_degrees = min(deg_lat, deg_lon)

        # Divide box into grid of cells of approximately equal length and width
        # NOTE: Small size differences may exist due to rounding.
        cells: List[BoundingBox] = bbox.split_into_squares(
            size_in_degrees=Decimal(str(max_side_degrees))
        )

        # Locate POIs within each cell if it contains any part of geography
        pois = []
        errors = []
        for cell in cells:
            if cell.intersects_with(geo):
                cell_pois, cell_errs = self.find_places_in_bounding_box(
                    box=cell, search_radius=YelpClient.MAX_SEARCH_RADIUS_IN_METERS
                )
                pois.extend(cell_pois)
                errors.extend(cell_errs)

        # Clean POIs
        cleaned_pois = self.clean_places(pois, geo)

        return PlacesSearchResult(pois, cleaned_pois, errors)
