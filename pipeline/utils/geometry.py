"""Provides helper functions and classes related to geometries.
"""

# Standard library imports
from decimal import Context, Decimal, ROUND_HALF_UP
from typing import List, Union

# Third-party imports
from pydantic import BaseModel, Field, model_validator
from shapely import MultiPolygon, Polygon



class Coordinate(BaseModel):
    """Simple data struture for latitude-longitude coordinates in EPSG:3857.
    """

    lat: Decimal = Field(ge=-90, le=90)
    """The latitude (i.e., y-coordinate).
    """
    lon: Decimal = Field(ge=-180, le=180)
    """The longitude (i.e., x-coordinate).
    """

    def to_list(self, use_lat_lon: bool=True) -> List[Decimal]:
        """Converts the coordinate to a two-item list of decimals.

        Args:
            use_lat_lon (`bool`): A boolean indicating whether the
                list should be generated in "latitude, longitude"
                order. Defaults to `True`.

        Returns:
            (`list` of `float`): The x- and y- coordinates. 
        """
        return [self.lat, self.lon] if use_lat_lon else [self.lon, self.lat]


class BoundingBox(BaseModel):
    """Simple data struture for a bounding box based on EPSG:3857 coordinates.
    """

    min_x: Decimal = Field(ge=-180, le=180)
    """The minimum longitude (i.e., x-coordinate).
    """

    max_x: Decimal = Field(ge=-180, le=180)
    """The maximum longitude (i.e., x-coordinate).
    """

    min_y: Decimal = Field(ge=-90, le=90)
    """The minimum latitude (i.e., y-coordinate).
    """

    max_y: Decimal = Field(ge=-90, le=90)
    """The maximum latitude (i.e., y-coordinate).
    """

    @property
    def top_left(self) -> Coordinate:
        """The top-left coordinate.
        """
        return Coordinate(lat=self.max_y, lon=self.min_x)

    @property
    def top_right(self) -> Coordinate:
        """The top-right coordinate.
        """
        return Coordinate(lat=self.max_y, lon=self.max_x)

    @property
    def bottom_left(self) -> Coordinate:
        """The bottom-left coordinate.
        """
        return Coordinate(lat=self.min_y, lon=self.min_x)

    @property
    def bottom_right(self) -> Coordinate:
        """The bottom-right coordinate.
        """
        return Coordinate(lat=self.min_y, lon=self.max_x)
    
    @property
    def width(self) -> float:
        """The width of the bounding box.
        """
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        """The height of the bounding box.
        """
        return self.max_y - self.min_y
    
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


def create_bbox_subdivisions(
    geo: Union[Polygon, MultiPolygon], 
    num_bbox: int) -> List[BoundingBox]:
    """Divides the bounding box of the given geometry
    into a list of smaller bounding boxes.

    Args:
        geo (`Polygon` or `MultiPolygon`): The geometry.

        num_bbox (`int`): The number of bounding boxes to create.

    Returns:
        (`list` of `BoundingBox`): The bounding boxes.
    """
    # Validate geography
    if (
        geo is None or 
        (not isinstance(geo, MultiPolygon) and not isinstance(geo, Polygon))
    ):
        raise ValueError("Expected a Shapely Polygon or Multipolygon.")
    
    # Initialize rounding
    context = Context(rounding=ROUND_HALF_UP)
    
    # Define dimensions of "global" bounding box for geography
    min_x, min_y, max_x, max_y = geo.bounds
    global_width = max_x - min_x
    global_length = global_width / num_bbox

    # Subdivide geography bounding box into several smaller "slices"
    slices = []
    for i in range(num_bbox):
        slice_min_x = min_x + (i * global_length)
        slice_max_x = slice_min_x + global_length
        slices.append(BoundingBox(
            min_x=round(Decimal(slice_min_x, context), 6), 
            max_x=round(Decimal(slice_max_x, context), 6), 
            min_y=round(Decimal(min_y, context), 6), 
            max_y=round(Decimal(max_y, context), 6)
        ))

    return slices
