"""Provides access to geographic locations using the TomTom Search API.
"""

# Standard library imports
import gzip
import json
import logging
import os
from enum import Enum
from typing import Dict, List, Union

# Third-party imports
import requests
from shapely import MultiPolygon, Polygon

# Application imports
from pipeline.scrape.common import IPlacesProvider
from pipeline.utils.geometry import create_bbox_subdivisions


class TomTomPOICategories(Enum):
    """Enumerates all relevant categories for points of interest.
    """
    # Potential Indoor Points
    CAFE_PUB = 9376
    RESTAURANT = 7315

    # Potential Outdoor Points
    AMUSEMENT_PARK = 9902
    COLLEGE_OR_UNIVERSITY = 7377
    CLUB_OR_ASSOCIAION = 9937
    HOTEL_OR_MOTEL = 7314
    LEISURE_CENTER = 9378
    MARKET = 7332
    MOVIE_THEATER = 7342
    NIGHTLIFE = 9379
    PARK = 9362
    PHARMACY = 7326
    PUBLIC_AMENITY = 9932
    PUBLIC_TRANSIT_STOP = 9942
    SCHOOL = 7372
    SHOP = 9361
    SHOPPING_CENTER = 7373
    THEATER = 7318
    TOURIST_ATTRACTION = 7376
    WORSHIP = 7339


class TomTomSearchClient(IPlacesProvider):
    """A simple wrapper for the TomTom Search API.
    """

    MAX_NUM_ASYNC_BATCH_REQUESTS = 10_000
    """The maximum number of API requests that can be sent
    at once using TomTom's asynchronous batch feature. 
    """ 

    MAX_NUM_RESULTS = 100
    """The maximum number of results that can be returned from a query.
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
                f"Missing expected environment variable \"{e}\"."
            ) from None

    def submit_async_batch(self, batch_items: List[Dict]) -> str:
        """Submits a new batch for asynchronous processing. Responds
        with a redirect to the location where the batch results
        can be obtained when the batch processing has completed.

        Args:
            batch_items (`list` of `dict`): The queries to send to
                other Search API endpoints.

        Documentation:
        - ["Asynchronous Batch Submission"](https://developer.tomtom.com/batch-search-api/documentation/asynchronous-batch-submission)
        
        Returns:
            (`str`): The download URL.
        """
        # Compose URL and request body
        url = f"https://api.tomtom.com/search/2/batch.json?key={self._api_key}"
        body = { "batchItems": batch_items }

        # Send HTTP POST request
        r = requests.post(url, json=body)
        if not r.ok:
            error = r.json()["error"]
            raise RuntimeError(
                "Failed to submit an asynchronous geometry search query "
                f"to the TomTom API. Received a \"{r.status_code}-{r.reason}\" "
                f"status code with the text: {error['description']}\"."
            )
        
        # Return batch id from response headers
        batch_id = r.headers.get("Location")
        if not batch_id:
            raise RuntimeError(
                "An unexpected error occurred. Could not parse download "
                "location from HTTP response headers."
        )

        return batch_id

    def download_async_batch(self, url: str) -> Dict:
        """Waits until asynchronous batch processing has 
        completed using a blocking, long poll request and
        then downloads the results.

        Documentation:
        - ["Asynchronous Batch Download"](https://developer.tomtom.com/batch-search-api/documentation/asynchronous-batch-download)

        Args:
            url (`str`): The URL to the download location.

        Returns:
            (`dict`): The response body.
        """
        while True:
            # Attempt to download results
            self._logger.info(
                "Calling TomTom Asynchronous Batch API to download results."
            )
            headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
            r = requests.get(url, headers=headers)

            # Raise exception if batch download request failed
            if not r.ok:
                error = r.json()["detailedError"]
                raise RuntimeError(
                    "Failed to download the results of an asynchronous "
                    f"geometry search query to the TomTom API. Received a "
                    f"\"{r.status_code}-{r.reason}\" status code with the text:"
                    f"\"{error['code']} - {error['details']['message']}\"."
                )
            
            # Raise exception if unanticipated success status occurs
            if r.status_code not in (200, 201):
                raise RuntimeError(
                    "Failed to download the results of an asynchronous "
                    f"geometry search query to the TomTom API. Received a "
                    f"\"{r.status_code}-{r.reason}\" status code unexpectedly."
                )

            # Decompress result and load as JSON if request succeeded
            if r.status_code == 200:
                self._logger.info("Data ready for download.")
                return json.load(gzip.decompress(r.text))
            
            # Otherwise, parse new batch id from location headers
            batch_id = r.headers.get("Location")

            # Rebuild URL
            url = (
                "https://api.tomtom.com/search/2/batch/"
                f"{batch_id}?key={self._api_key}"
            )
                          
    def find_places_in_geography(
        self, 
        geo: Union[Polygon, MultiPolygon]) -> List[Dict]:
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
        # Divide geography into boxes corresponding to separate POI queries
        try:
            num_bboxes = TomTomSearchClient.MAX_NUM_ASYNC_BATCH_REQUESTS
            self._logger.info(f"Subdiving geography into {num_bboxes} box(es).")
            bboxes = create_bbox_subdivisions(geo, num_bboxes)
        except Exception as e:
            raise RuntimeError(
                "Failed to query TomTom API for points of interest. "
                "An unexpected error occurred while dividing the given "
                f"geography into smaller boxes. \"{e}\""
            ) from e

        # Construct query for each box
        self._logger.info("Creating POI search queries for each box.")
        batch_items = []
        categories = ",".join(e.value for e in TomTomPOICategories)
        for bbox in bboxes:
            batch_items.append({
                "query": (
                    "/poiSearch/.json"
                    f"limit={TomTomSearchClient.MAX_NUM_RESULTS}"
                    f"&categorySet={categories}"
                    f"&openingHours=nextSevenDays"
                    f"&topLeft={bbox.top_left.to_list(use_lat_lon=False)}"
                    f"&btmRight={bbox.bottom_right.to_list(use_lat_lon=False)}"
                )
            })

        # Submit queries to API in batch
        try:
            self._logger.info("Submitting POI queries in batch to API.")
            url = self.submit_async_batch(batch_items)
        except Exception as e:
            raise RuntimeError(
                "Failed to query TomTom API for points of interest. "
                "An unexpected error occurred while submitting "
                f"queries in batch. \"{e}\""
            ) from e
        
        # Download results
        try:
            self._logger.info("Downloading results of batch search.")
            results = self.download_async_batch(url)
            self._logger.info("Download complete.")
        except Exception as e:
            raise RuntimeError(
                "Failed to download results of TomTom API Asynchronous "
                f"Batch Search for points of interest. \"{e}\""
            ) from e

        # Log outcome
        total = results["summary"]["totalRequests"]
        num_successes = results["summary"]["successfulRequests"]
        self._logger.info(
            f"Of {total} total request(s), {num_successes} "
            f"succeeded and {total - num_successes} failed."
        )

        # Parse results
        data = []
        errors = []
        for item in results["batchItems"]:
            if item["statusCode"] == 200:
                data.extend(item["results"])
            else:
                errors.append(item)

        # TODO: Handle errors
        # TODO: Clip results to original boundary
        
        return data
