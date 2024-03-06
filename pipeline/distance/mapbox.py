"""Utilities for computing distances between coordinates.
"""

# Standard library imports
import os
import time
from enum import Enum
from typing import List

# Third-party imports
import requests
import tqdm

# Application imports
from pipeline.utils.geometry import WGS84Coordinate
from pydantic import BaseModel


class MapboxRouteEnum(Enum):
    """Enumerates Mapbox Matrix API routing profiles."""

    CYCLING = "mapbox/cycling"
    DRIVING = "mapbox/driving"
    DRIVING_TRAFFIC = "mapbox/driving-traffic"
    WALKING = "mapbox/walking"


class MapboxRoutingProfile(BaseModel):
    """Represents a mapbox routing profile."""

    name: MapboxRouteEnum
    """The routing name (e.g., "mapbox/driving").
    """

    @property
    def max_coords(self) -> int:
        """The maximum number of coordinates permitted for matrix operations.
        According to the documentation, "mapbox/driving-traffic" can send
        up to 10 coordinates to the server per request while all other profiles
        can send up to 25 coordinates at a time.
        """
        return 10 if self.name == MapboxRouteEnum.DRIVING_TRAFFIC else 25

    @property
    def max_requests_per_minute(self) -> int:
        """The maximum number of requests permitted for the profile per minute."""
        return 30 if self.name == MapboxRouteEnum.DRIVING_TRAFFIC else 60


class MapboxApiClient:
    """A simple wrapper for the Mapbox API."""

    def __init__(self) -> None:
        """Initializes a new instance of a `MapboxApiClient`.

        Raises:
            `RuntimeError` if missing the environment variable `MAPBOX_API_KEY`.

        Args:
            `None`

        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["MAPBOX_API_KEY"]
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize MapboxApiClient."
                f'Missing expected environment variable "{e}".'
            ) from None

    def make_distance_matrix(
        self, coords: List[WGS84Coordinate], profile: MapboxRoutingProfile
    ) -> List[List[float]]:
        """Calculates the fastest route between each unique pair of coordinates
        given the routing profile (e.g., driving, walking, cycling) and then
        generates a matrix of distance traveled in meters for each route.

        Documentation:
            - ["Routing profiles"](https://docs.mapbox.com/api/navigation/directions/#routing-profiles)
            - ["Retrieve a matrix"](https://docs.mapbox.com/api/navigation/matrix/#retrieve-a-matrix)

        Args:
            coords (`list` of `Coordinate`): The latitude-longitude pairs.

            profile (`MapboxRoutingProfile`): The mode of travel.

        Returns:
            (`list` of `list` of `float`): Distances as an array of
                arrays that represent the matrix in row-major order.
                `distances[i][j]` gives the travel distance from the ith
                source to the jth destination. All values are in meters.
                The distance between the same coordinate is always 0.
                Finding no distance, the result will be null.
        """
        # Initialize variables
        matrix = []
        batch_size = profile.max_coords
        seconds_delay = 60 / profile.max_requests_per_minute

        # Populate the matrix by iterating through every coordinate
        # (source) and computing its distance to every other
        # coordinate (destination), including itself.
        for i in tqdm.tqdm(range(len(coords))):
            row = []
            for j in range(0, len(coords), batch_size):
                # Compose request
                batch = [coords[i]] + coords[j : j + batch_size]
                coord_str = ";".join(f"{coord.lon},{coord.lat}" for coord in batch)
                dest_indices = ";".join(str(i) for i in range(1, len(batch)))
                url = (
                    f"https://api.mapbox.com/directions-matrix/v1/"
                    f"{profile.name.value}/{coord_str}"
                )
                params = {
                    "access_token": self._api_key,
                    "annotations": "distance",
                    "sources": "0",
                    "destinations": dest_indices,
                }

                # Make request
                r = requests.get(url, params=params)
                if not r.ok:
                    raise RuntimeError(
                        f"Error fetching distances from the Mapbox API. "
                        f'The server returned a "{r.status_code}-{r.reason}" '
                        f'status code with the text: "{r.text}".'
                    )

                # Process the JSON response
                distances = r.json()["distances"][0]
                row.extend(distances)
                time.sleep(seconds_delay)

            # Append completed row to matrix
            matrix.append(row)

        return matrix
