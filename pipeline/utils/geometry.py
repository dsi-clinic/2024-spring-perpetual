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


class BoundingBox(BaseModel):
    """Simple data struture for a bounding box based on EPSG:4326 coordinates."""

    min_x: Decimal = Field(ge=-180, le=180)
    """The minimum longitude (i.e., x-coordinate).
    """

    max_x: Decimal = Field(ge=-180, le=180)
    """The maximum longitude (i.e., x-coordinate).`
    """

    min_y: Decimal = Field(ge=-90, le=90)
    """The minimum latitude (i.e., y-coordinate).
    """

    max_y: Decimal = Field(ge=-90, le=90)
    """The maximum latitude (i.e., y-coordinate).
    """

    @property
    def top_left(self) -> WGS84Coordinate:
        """The top-left coordinate."""
        return WGS84Coordinate(lat=self.max_y, lon=self.min_x)

    @property
    def top_right(self) -> WGS84Coordinate:
        """The top-right coordinate."""
        return WGS84Coordinate(lat=self.max_y, lon=self.max_x)

    @property
    def bottom_left(self) -> WGS84Coordinate:
        """The bottom-left coordinate."""
        return WGS84Coordinate(lat=self.min_y, lon=self.min_x)

    @property
    def bottom_right(self) -> WGS84Coordinate:
        """The bottom-right coordinate."""
        return WGS84Coordinate(lat=self.min_y, lon=self.max_x)

    @property
    def center(self) -> WGS84Coordinate:
        """The center point of the bounding box."""
        return WGS84Coordinate(
            lat=(self.min_y + self.max_y) / 2, lon=(self.min_x + self.max_x) / 2
        )

    @property
    def width(self) -> Decimal:
        """The width of the box in degrees."""
        return self.max_x - self.min_x

    @property
    def height(self) -> Decimal:
        """The width of the box in degrees."""
        return self.max_y - self.min_y

    @classmethod
    def from_polygon(
        cls, polygon: Union[MultiPolygon, Polygon]
    ) -> "BoundingBox":
        """Creates a new `BoundingBox` instance from the
        minimum bounding region of a polygon.

        Raises:
            `TypeError` if the polygon is not an instance of
                a Shapely `Polygon` or `MultiPolygon`.

        Args:
            polygon (`Polygon` or `MultiPolygon`): The polygon.

        Returns:
            (`BoundingBox`): The bounding box.
        """
        if (not isinstance(polygon, MultiPolygon)) and (
            not isinstance(polygon, Polygon)
        ):
            raise TypeError("Expected a Shapely Polygon or Multipolygon.")
        min_x, min_y, max_x, max_y = polygon.bounds
        return BoundingBox(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

    def intersects_with(self, polygon: Union[MultiPolygon, Polygon]) -> bool:
        """Determines whether the bounding box intersects with
        the given polygon. An intersection occurs when two shapes
        at least one boundary or interior point in common.

        Documentation:
        - ["shapely.overlaps — Shapely 2.0.3 Documentation"](https://shapely.readthedocs.io/en/stable/reference/shapely.overlaps.html)

        Args:
            polygon (`Polygon` or `MultiPolygon`): The polygon.

        Returns:
            (`bool`): `True` if there is an intersection and `False` otherwise.s
        """
        shapely_repr = box(
            float(self.min_x),
            float(self.min_y),
            float(self.max_x),
            float(self.max_y),
        )
        return intersects(polygon, shapely_repr)

    def split_along_axes(self, x_into: int, y_into: int) -> List["BoundingBox"]:
        """Splits the bounding box into a given number of units in
        the x- and y-directions to create a list of smaller bounding boxes.
        For example, the arguments `x_into=2` and `y_into=3` would
        divide the bounding box into two pieces along the x-axis and
        three pieces along the y-axis, for a total of six smaller boxes,
        each of which would be half of the original width and one-third of
        the original height.

        Raises:
            `ValueError` if `x_into` or `y_into` is less than or equal to zero.

        Args:
            x_into (`int`): The number of pieces the bounding box should
                be split into along the x-axis.

            y_into (`int`): The number of pieces the bounding box should
                be split into along the y-axis.

        Returns:
            (`list` of `BoundingBox`): The bounding boxes.
        """
        # Validate arguments
        if (x_into <= 0) or (y_into <= 0):
            raise ValueError(
                "Unable to split bounding box. Expected the number of slices "
                "to make along the x- and y-axes to be positive numbers."
            )

        # Initialize rounding strategy to prevent long decimals
        context = Context(rounding=ROUND_HALF_UP)

        # Define dimensions of bounding box "slices"
        slice_width = self.width / x_into
        slice_height = self.height / y_into

        # Subdivide bounding box into slices
        slices = []
        for i in range(x_into):
            slice_min_x = self.min_x + (i * slice_width)
            slice_max_x = slice_min_x + slice_width
            for j in range(y_into):
                slice_min_y = self.min_y + (j * slice_height)
                slice_max_y = slice_min_y + slice_height
                slices.append(
                    BoundingBox(
                        min_x=round(Decimal(slice_min_x, context), 6),
                        max_x=round(Decimal(slice_max_x, context), 6),
                        min_y=round(Decimal(slice_min_y, context), 6),
                        max_y=round(Decimal(slice_max_y, context), 6),
                    )
                )

        return slices

    def split_into_squares(
        self, size_in_degrees: Decimal
    ) -> List["BoundingBox"]:
        """Splits the bounding box into squares of the given size in degrees.
        If the bounding box cannot be divided into squares, its dimensions
        are extended until the operation is possible.

        Raises:
            `ValueError` if `size_in_degrees` is less than or equal to zero.

        Args:
            size_in_degrees (`float`): The number of degrees latitude
                and longitude for each square.

        Returns:
            (`list` of `BoundingBox`): The squares.
        """
        # Validate arguments
        if size_in_degrees <= 0:
            raise ValueError(
                "Unable to split bounding box. Expected a positive "
                "number for the size of the square subdivisions."
            )

        # Initialize rounding strategy to prevent long decimals
        context = Context(rounding=ROUND_HALF_UP)

        # Coerce bounding box into square shape
        longest_side = max(self.width, self.height)
        max_x = self.min_x + longest_side
        max_y = self.min_y + longest_side
        bbox = BoundingBox(
            min_x=self.min_x, max_x=max_x, min_y=self.min_y, max_y=max_y
        )

        # Determine number of rows/columns necessary for sub-squares of equal size
        dim = math.ceil(bbox.height / size_in_degrees)

        # Subdivide bounding box into squares
        squares = []
        for i in range(dim):
            slice_min_x = float(self.min_x + (i * size_in_degrees))
            slice_max_x = slice_min_x + float(size_in_degrees)
            for j in range(dim):
                slice_min_y = float(self.min_y + (j * size_in_degrees))
                slice_max_y = slice_min_y + float(size_in_degrees)
                squares.append(
                    BoundingBox(
                        min_x=round(Decimal(slice_min_x, context), 6),
                        max_x=round(Decimal(slice_max_x, context), 6),
                        min_y=round(Decimal(slice_min_y, context), 6),
                        max_y=round(Decimal(slice_max_y, context), 6),
                    )
                )

        return squares

    @model_validator(mode="after")
    def validate_coords(self) -> None:
        """Validates the set of coordinates to confirm
        correct relationships between them.

        Args:
            `None`

        Returns:
            `None`
        """
        if self.min_x == self.max_x or self.min_y == self.max_y:
            raise ValueError("The bounding box must be two-dimensional.")
        if self.min_x > self.max_x:
            raise ValueError(
                "The min x-coordinate is greater than the max x-coordinate."
            )
        if self.min_y > self.max_y:
            raise ValueError(
                "The min y-coordinate is greater than the max y-coordinate."
            )


def convert_meters_to_degrees(
    meters: float, coord: WGS84Coordinate
) -> Tuple[float, float]:
    """Converts a distance expressed in meters into
    units of degrees at a point on the WGS 84 datum.

    Args:
        meters (`float`): The distance.

        coord (`WGS84Coordinate`): The latitude-longitude coordinate.

    Returns:
        ((`float`, `float`)): A two-item tuple consisting of the
            degrees latitude and longitude of the converted
            distance, respectively.
    """
    # Convert distance for latitude
    degrees_lat = meters / 111000

    # Convert distance for longitude
    latitude_radians = math.radians(float(coord.lat))
    degrees_lon = meters / (111000 * math.cos(latitude_radians))

    return degrees_lat, degrees_lon
