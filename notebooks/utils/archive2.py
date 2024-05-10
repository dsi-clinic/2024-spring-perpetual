"""Provides helper functions and classes related to geometries.
"""

# Standard library imports
import math
from decimal import ROUND_HALF_UP, Context, Decimal
from typing import List, Tuple, Union

# Third-party imports
from pydantic import BaseModel, Field, model_validator
from shapely import MultiPolygon, Polygon, box, intersects


class WGS84Coordinate(BaseModel):
    """Simple data struture for a latitude-longitude coordinate in EPSG:4326."""

    lat: Decimal = Field(ge=-90, le=90)
    """The latitude (i.e., y-coordinate).
    """
    lon: Decimal = Field(ge=-180, le=180)
    """The longitude (i.e., x-coordinate).
    """

    def to_list(self, as_lat_lon: bool = True) -> List[Decimal]:
        """Converts the coordinate to a two-item list of decimals.

        Args:
            as_lat_lon (`bool`): A boolean indicating whether the
                list should be generated in "latitude, longitude"
                order. Defaults to `True`.

        Returns:
            (`list` of `float`): The x- and y- coordinates.
        """
        return [self.lat, self.lon] if as_lat_lon else [self.lon, self.lat]