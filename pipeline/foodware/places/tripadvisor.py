"""Provides access to large hotels using the TripAdvisor API.
"""

# Standard library imports
import os
import re
import time
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import haversine as hs
import requests
from bs4 import BeautifulSoup as soup

# Application imports
from common.geometry import BoundingBox
from common.logger import logging
from foodware.places.common import IPlacesProvider, Place, PlacesSearchResult
from shapely import MultiPolygon, Point, Polygon


class TripadvisorClient(IPlacesProvider):
    """Provides access to points of interest from TripAdvisor."""

    MAX_NUM_RESULTS_PER_REQUEST: int = 10
    """The maximum number of records that can be returned from a single request.
    """

    ROOT_URL: str = "https://www.tripadvisor.com"
    """The root URL of the Tripadvisor website.
    """

    LOCATION_DETAILS_URL: str = "https://api.content.tripadvisor.com/api/v1/location/{location_id}/details"
    """The base URL to use when searching for location details by id.
    """

    NEARBY_SEARCH_URL: str = (
        "https://api.content.tripadvisor.com/api/v1/location/nearby_search"
    )
    """The base URL to use when searching for locations within a circular area.
    """

    NEARBY_SEARCH_RADIUS_UNIT: str = "mi"
    """The unit of measurement to use when restricting nearby location
    searches to a radius around a latitude-longitude point. Defaults
    to "mi" for miles.
    """

    TEXT_SEARCH_URL: str = (
        "https://api.content.tripadvisor.com/api/v1/location/search"
    )
    """The base URL to use when searching for locations using a text query.
    """

    SECONDS_DELAY_PER_REQUEST: float = 0.5
    """The number of seconds to wait after each HTTP request.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `TripadvisorClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if the environment variables
                `TRIPADVISOR_API_KEY` and
                `PROXYSCRAPE_API_KEY` are not found.

        Returns:
            `None`
        """
        try:
            self._tripadvisor_api_key = os.environ["TRIPADVISOR_API_KEY"]
            self._proxyscrape_api_key = os.environ["PROXYSCRAPE_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GooglePlacesClient."
                f'Missing expected environment variable "{e}".'
            ) from None

    def clean_places(
        self, places: List[Dict], geo: Union[MultiPolygon, Polygon]
    ) -> List[Dict]:
        """Cleans a list of places newly fetched from the API
        by annotating them with additional details and removing
        duplicates and those that lie outside the geography.

        Args:
            places (`list` of `dict`): The places to clean.

            geo (`shapely.MultiPolygon`|`shapely.Polygon`): The geography boundary.

        Returns:
            (`list` of `dict`): The cleaned places.
        """
        # Initialize variables
        visited_ids = set()
        cleaned_places = []

        # Process each raw place
        for place in places:
            # Extract id
            id = place["location_id"]

            # Skip processing if already visited
            if id in visited_ids:
                continue

            # Otherwise, fetch and parse place details
            details = self.get_location_details(id)
            lon = float(details["longitude"])
            lat = float(details["latitude"])

            # Terminate processing if place outside geography
            if geo and not geo.contains(Point(lon, lat)):
                visited_ids.add(id)
                continue

            # Otherwise, fetch place room count
            details["room_count"] = self.get_room_count(details["web_url"])

            # Map place to standard format and append to list
            mapped = self.map_place(details)
            cleaned_places.append(vars(mapped))

            # Mark place as seen
            visited_ids.add(id)

        return cleaned_places

    def get_location_details(self, id: int) -> Dict:
        """Queries the Tripadvisor API for details about a location.

        Args:
            id (`int`): The unique identifier for the location, assigned by Tripadvisor.

        Returns:
            (`dict`): The location response.
        """
        # Add delay before request
        time.sleep(TripadvisorClient.SECONDS_DELAY_PER_REQUEST)

        # Define local function to send location details request to Tripadvisor API
        def send_request():
            url = TripadvisorClient.LOCATION_DETAILS_URL.format(location_id=id)
            headers = {"accept": "application/json"}
            api_params = {
                "key": self._tripadvisor_api_key,
                "language": "en",
                "currency": "USD",
            }
            return requests.get(url, headers=headers, params=api_params)

        # Define local function to handle retries for transient connectivity issues
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

        # Send request
        r = retry(send_request)

        # Check for authentication error, in which case
        # processing should end immediately
        if r.status_code == 403:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API key is likely '
                "missing this device's IP address. Head to "
                "https://www.tripadvisor.com/developers?screen=credentials to "
                "add this device's IP address to the API key."
            )

        # Check daily limit for searches reached, in which
        # case processing should end immediately
        if r.status_code == 429:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API has reached '
                "its daily limit of requests. Please try again later."
            )

        return r.json()

    def get_room_count(self, tripadvisor_url: str) -> Optional[int]:
        """Scrapes a Tripadvisor location review webpage for the number of rooms.
        Valid for locations categorized as hotels only, and not all pages are
        expected to contain this information.

        Args:
            tripadvisor_url (`str`): The URL to the webpage.

        Returns:
            (`int` | `None`): The room count, if one exists.
        """
        # Initialize request parameters
        url = "https://api.proxyscrape.com/v3/accounts/freebies/scraperapi/request"
        data = {"url": tripadvisor_url, "browserHtml": True}
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self._proxyscrape_api_key,
        }

        # Send request
        time.sleep(TripadvisorClient.SECONDS_DELAY_PER_REQUEST)
        r = requests.post(url, headers=headers, json=data)

        # Handle error
        if not r.ok:
            raise RuntimeError(
                f'The request to fetch the URL "{url}" failed with '
                f'a "{r.status_code}-{r.reason}" status code and '
                f'the message "{r.text}".'
            )

        # Otherwise, parse response
        html = r.json()["data"]["browserHtml"]
        s = soup(html, features="lxml")
        try:
            room_count_div = s.find("div", text="NUMBER OF ROOMS").find_next(
                "div"
            )
            return int(room_count_div.text.strip())
        except (AttributeError, ValueError):
            return None

    def map_place(self, place: Dict) -> Place:
        """Maps a place fetched from a data source to a standard representation.

        Args:
            place (`dict`): The place.

        Returns:
            (`Place`): The standardized place.
        """
        id = place["location_id"]
        name = place["name"]
        categories = place["category"]["name"]
        aliases = place["category"]["name"]
        lat = float(place["latitude"])
        lon = float(place["longitude"])
        address = place["address_obj"]["address_string"]
        is_closed = False
        source = "tripadvisor"
        features = {"room_count": place["room_count"]}

        return Place(
            id,
            name,
            categories,
            aliases,
            lat,
            lon,
            address,
            is_closed,
            source,
            features=features,
        )

    def find_places_in_bounding_box(
        self,
        original_geo,
        box: BoundingBox,
        category: str,
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
        # Define request parameters
        api_params = {
            "latLong": f"{float(box.center.lat)},{float(box.center.lon)}",
            "radius": search_radius,
            "radiusUnit": TripadvisorClient.NEARBY_SEARCH_RADIUS_UNIT,
            "category": category,
            "key": self._tripadvisor_api_key,
            "language": "en",
        }

        # Define local function to send nearby search request to Tripadvisor API
        def send_request():
            url = TripadvisorClient.NEARBY_SEARCH_URL
            headers = {"accept": "application/json"}
            return requests.get(url, headers=headers, params=api_params)

        # Define local function to handle retries for transient connectivity issues
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

        # Send request
        r = retry(send_request)

        # Check for authentication error, in which case
        # processing should end immediately
        if r.status_code == 403:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API key is likely '
                "missing this device's IP address. Head to "
                "https://www.tripadvisor.com/developers?screen=credentials to "
                "add this device's IP address to the API key."
            )

        # Check daily limit for searches reached, in which
        # case processing should end immediately
        if r.status_code == 429:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API has reached '
                "its daily limit of requests. Please try again later."
            )

        # Otherwise, sleep and then parse JSON from response body
        try:
            time.sleep(TripadvisorClient.SECONDS_DELAY_PER_REQUEST)
            payload = r.json()
        except Exception as e:
            self._logger.error(f"Failed to parse reponse body JSON. {e}")
            return [], [{"api_params": api_params, "error": str(e)}]

        # If error occurred, store information and exit processing for cell
        if not r.ok or "error" in payload:
            self._logger.error(
                "Failed to retrieve POI data through the Tripadvisor API. "
                f'Received a "{r.status_code}-{r.reason}" status code '
                f'with the message "{r.text}".'
            )
            return [], [{"api_params": api_params, "error": payload}]

        # Otherwise, if no data returned, return empty lists of POIs and errors
        if "data" not in payload:
            self._logger.warning("No data found in response body.")
            return [], []

        # Otherwise, if number of POIs returned equals max,
        # split box and recursively issue HTTP requests
        if (
            len(payload["data"])
            >= TripadvisorClient.MAX_NUM_RESULTS_PER_REQUEST
        ):
            pois, errors = [], []
            sub_cells = box.split_along_axes(x_into=2, y_into=2)
            for sub in sub_cells:
                if sub.intersects_with(original_geo):
                    sub_hotels, sub_errs = self.find_places_in_bounding_box(
                        original_geo, sub, category, search_radius / 2
                    )
                    pois.extend(sub_hotels)
                    errors.extend(sub_errs)
            return pois, errors

        # Otherwise, return locations
        return payload["data"], []

    def run_nearby_search(
        self, geo: Union[Polygon, MultiPolygon], categories: List[str]
    ) -> PlacesSearchResult:
        """Locates all POIs within the given geography and maps them to
        a standard place representation. The Tripadvisor API permits searching
        for POIs within an area around a given point. Therefore, data is
        extracted by dividing the geography's bounding box into one or more cells of
        equal size and then searching within the circular areas that circumscribe
        (i.e., perfectly enclose) those cells.

        To circumscribe a cell, the circle must have a radius that is one-half the
        length of the cell's diagonal (as derived from the Pythagorean Theorem).
        Let `side` equal the length of a cell's side. It follows that the radius is:

        ```
        radius = (√2/2) * side
        ```

        Because the Tripadvisor API does not enforce an upper limit on the search
        radius, this circumscribed circle's radius is used directly as an API
        request parameter without further transformation.

        Finally, at the time of writing, a maximum of 10 records are returned per
        search query, even if more locations are available. Therefore, the function
        recursively searches in smaller and smaller areas if 10 records are encountered
        in the response payload to avoid missing data. The Tripadvisor API is also
        quirky in that only locations whose subcategory, not primary category, is
        "hotels" is available through the API when using the "category" query parameter.
        This removes many small bed and breakfast locations that can otherwise be
        viewed on the Tripadvisor website.

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`PlacesSearchResult`): The result of the geography query.
                Contains a raw list of retrieved places, a list of
                cleaned places, and a list of any errors that occurred.
        """
        # Calculate bounding box for geography
        bbox: BoundingBox = BoundingBox.from_polygon(geo)

        # Calculate radius of circle that circumscribes box, in miles
        top_left_pt = tuple(bbox.top_left.to_list(coerce_to_float=True))
        bottom_right_pt = tuple(bbox.bottom_right.to_list(coerce_to_float=True))
        diagonal_length = hs.haversine(
            top_left_pt, bottom_right_pt, hs.Unit.MILES
        )
        search_radius = round(diagonal_length / 2)

        # Locate all POIs within bounding box for each specified category
        pois = []
        errors = []
        for category in categories:
            cat_pois, cat_errors = self.find_places_in_bounding_box(
                original_geo=geo,
                box=bbox,
                category=category,
                search_radius=search_radius,
            )
            pois.extend(cat_pois)
            errors.extend(cat_errors)

        # Clean and map POIs
        cleaned_pois = self.clean_places(pois, geo)

        return PlacesSearchResult(pois, cleaned_pois, errors)

    def run_text_search(
        self,
        query: str,
        restriction: Polygon | MultiPolygon | None = None,
        category: str = "hotels",
    ) -> PlacesSearchResult:
        """Locates all POIs matching the given text query and contained by
        the given geography and maps them to a standard place representation.

        Args:
            query (`str`): The query used to filter locations (e.g., "Hilo, Hawaii").

            restriction (`Polygon` or `MultiPolygon`): An optional boundary,
                used to clip fetched search results. Defaults to `None`.

            category (`str`): The category used to filter search results.
                Choices include "hotels", "attractions", "restaurants",
                and "geos". Defaults to "hotels".

        Returns:
            (`PlacesSearchResult`): The result of the geography query.
                Contains a raw list of retrieved places, a list of
                cleaned places, and a list of any errors that occurred.
        """
        # Format search query by standardizing case, stripping whitespace, URL encoding
        try:
            clean_query = query.strip().replace(" ", "%20").lower()
        except Exception as e:
            raise RuntimeError(f'Error cleaning search query "{query}". {e}')

        # Search Tripadvisor for locations affiliated with query
        url = TripadvisorClient.TEXT_SEARCH_URL
        headers = {"accept": "application/json"}
        api_params = {
            "key": self._tripadvisor_api_key,
            "searchQuery": clean_query,
            "category": "geos",
            "language": "en",
        }
        r = requests.get(url, headers=headers, params=api_params)

        # Check for authentication error, in which case
        # processing should end immediately
        if r.status_code == 403:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API key is likely '
                "missing this device's IP address. Head to "
                "https://www.tripadvisor.com/developers?screen=credentials to "
                "add this device's IP address to the API key."
            )

        # Check daily limit for searches reached, in which
        # case processing should end immediately
        if r.status_code == 429:
            raise requests.exceptions.HTTPError(
                "The request to the Tripadvisor API failed with a "
                f'"{r.status_code} - {r.reason}" status code and the '
                f'message "{r.text}". The Tripadvisor API has reached '
                "its daily limit of requests. Please try again later."
            )

        # Otherwise, parse data
        payload = r.json()

        # If request failed, return error
        if not r.ok or "error" in payload:
            self._logger.error(
                "Failed to retrieve POI data through the Tripadvisor API. "
                f'Received a "{r.status_code}-{r.reason}" status code '
                f'with the message "{r.text}".'
            )
            return PlacesSearchResult(
                raw=[],
                clean=[],
                errors=[{"api_params": api_params, "error": payload["error"]}],
            )

        # If no data present in response payload, return error
        if not len(payload["data"]):
            return PlacesSearchResult(
                raw=[],
                clean=[],
                errors=[
                    {
                        "api_params": api_params,
                        "error": (
                            "The Tripadvisor API returned no results "
                            "for the location search query."
                        ),
                    }
                ],
            )

        # Otherwise, parse fields from top search result
        top_result = payload["data"][0]
        location_id = top_result["location_id"]
        name = top_result["name"]
        secondary_name = top_result["address_obj"].get("city", "")
        state = top_result["address_obj"]["state"]

        # Build URL to starting search results page
        if category == "hotels":
            entity = "Hotels"
        else:
            raise ValueError(
                "Only the hotels category is currently configured to be"
                " scraped."
            )
        search_page_url = (
            f"{TripadvisorClient.ROOT_URL}/{entity}-g{location_id}-{name}"
            f"{'_' if secondary_name else ''}{secondary_name}_{state}-{entity}.html"
        )
        search_page_url = "_".join(search_page_url.split())

        # Crawl through results pages to get ids of all locations
        location_ids = self.scrape_results_pages(search_page_url, category)

        # Build and clean locations
        locations = [{"location_id": id} for id in location_ids]
        cleaned_locations = self.clean_places(locations, restriction)

        # Return search results
        return PlacesSearchResult(raw=[], clean=cleaned_locations, errors=[])

    def scrape_results_pages(
        self, starting_url: str, category: str
    ) -> List[int]:
        """Scrapes Tripadvisor results pages for a search query
        to retrieve a list of the results' location ids.

        Args:
            starting_url (`str`): The first page of the search results.

            category (`str`): The category being searched for.

        Returns:
            (`list` of `int`): The location ids.
        """
        # Initialize variables
        location_ids = []
        current_search_results_url = starting_url

        # Validate category
        if category == "hotels":
            entity = "hotel"
        else:
            raise ValueError(
                "Only the hotels category is currently configured to be"
                " scraped."
            )

        while True:
            # Initialize request parameters
            url = "https://api.proxyscrape.com/v3/accounts/freebies/scraperapi/request"
            data = {"url": current_search_results_url, "browserHtml": True}
            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": self._proxyscrape_api_key,
            }

            # Send request
            time.sleep(TripadvisorClient.SECONDS_DELAY_PER_REQUEST)
            r = requests.post(url, headers=headers, json=data)

            # Handle error if non-success response received
            if not r.ok:
                raise RuntimeError(
                    f'The request to fetch the search results page "{url}" '
                    f'failed with a "{r.status_code}-{r.reason}" status '
                    f'code and the message "{r.text}".'
                )

            # Otherwise, parse response body
            html = r.json()["data"]["browserHtml"]
            s = soup(html, features="lxml")

            # Get links to all hotels
            hotel_cards = s.find_all(
                "div", {"data-automation": f"{entity}-card-title"}
            )
            for card in hotel_cards:
                hotel_url = card.find("a")["href"]
                location_id = hotel_url.split("-")[2].strip("d")
                location_ids.append(int(location_id))

            # Stop crawling if last results page scraped
            pagination_text = s.find("div", class_="Ci").text
            numbers = list(map(int, re.findall(r"\d+", pagination_text)))
            interval_end_boundary = numbers[-2]
            total_city_results = numbers[-1]
            if interval_end_boundary >= total_city_results:
                return location_ids

            # Otherwise, build URL to next results page
            g_location_id = current_search_results_url.split("-")[1]
            if "_" in current_search_results_url.split("-")[2]:
                current_search_results_url = current_search_results_url.replace(
                    g_location_id, g_location_id + "-oa30"
                )
            else:
                oa_old = current_search_results_url.split("-")[2]
                oa_old_num = int(oa_old[2:])
                glocationid_oa = g_location_id + "-" + oa_old
                oa_new_num = oa_old_num + 30
                oa_new = "oa" + str(oa_new_num)
                current_search_results_url = current_search_results_url.replace(
                    glocationid_oa, g_location_id + "-" + oa_new
                )
