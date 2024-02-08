"""Provides helper functions and classes related to geometries.
"""

# Standard library imports
from decimal import Context, Decimal, ROUND_HALF_UP
from typing import List, Union

# Third-party imports
from pydantic import BaseModel, Field, model_validator
from shapely import MultiPolygon, Polygon
import geopandas as gpd
from shapely.geometry import Point


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


'''adding the functions needed to divide the given geojson file'''



'''the input to the get_bounding_box functin is the path to the
geojson file you're using for the API, the output is the max bounds of the
geo area, which will be used to make subdivisions'''

def get_bounding_box(geojson_file):
    '''Load GeoJSON file into GeoDataFrame to get the bounds'''
    gdf = gpd.read_file(geojson_file)

    '''Get the total bounding box of all features in the GeoDataFrame;
    The total bounding box is the area that encompasses all of the given 
    geographical area in the geojson file'''
    
    geojson_bbox = gdf.total_bounds

    '''Initialize min_lon, min_lat, max_lon, max_lat using 
    the bounding box of the GeoDataFrame, this will be used to
    create a large box which we will make sub-divisions from'''
    min_lon, min_lat, max_lon, max_lat = geojson_bbox

    '''maxing the values so that we make sure to stay as close to the
    the given geographical area'''
    total_bbox = (max(min_lon, geojson_bbox[0]), max(min_lat, geojson_bbox[1]),
                  min(max_lon, geojson_bbox[2]), min(max_lat, geojson_bbox[3]))

    return total_bbox


'''the second function of this set is using the output of get_bounding_box,
it takes in the bounds of the total area and divides them based on a given n_lon
and n_lat, 

n_long and n_lat are the number of divisions you want across the long and lat 
axes, for example if you want 18 square quadrants, you would do n_long,n_lat=9


the min_long...etc values can be saved into one variable and input like this
total_bbox=get_bounding_area(file) generate_quadrants(*total_bbox,9,9)'''

def generate_quadrants(min_lon, min_lat, max_lon, max_lat, n_lon, n_lat):
    midpoints = []
    quadrants = []
    '''creating quadrants using midpoints'''
    for i in range(n_lon):
        for j in range(n_lat):
            quad_min_lon = min_lon + i * (max_lon - min_lon) / n_lon
            quad_max_lon = min_lon + (i + 1) * (max_lon - min_lon) / n_lon
            quad_min_lat = min_lat + j * (max_lat - min_lat) / n_lat
            quad_max_lat = min_lat + (j + 1) * (max_lat - min_lat) / n_lat

            midpoint_lon = (quad_min_lon + quad_max_lon) / 2
            midpoint_lat = (quad_min_lat + quad_max_lat) / 2
            midpoints.append((midpoint_lon, midpoint_lat))

            '''corrects the bounds to make sure they're not outside the
            range of total bounds'''
            quad_min_lon = max(min_lon, min(quad_min_lon, max_lon))
            quad_max_lon = min(max_lon, max(quad_max_lon, min_lon))
            quad_min_lat = max(min_lat, min(quad_min_lat, max_lat))
            quad_max_lat = min(max_lat, max(quad_max_lat, min_lat))

            quadrant = (quad_min_lon, quad_max_lon, quad_min_lat, quad_max_lat)
            quadrants.append(quadrant)
    '''function returns quadrants and midpoints, midpoint coords
    needed to find centerpoints'''
    return midpoints, quadrants



'''next set of functions work together to calculate center point
of each quadrant

the input of the functions is the quadrants which we get from
generate_quadrants'''

def calculate_quadrant_center(quadrant):

    '''takes bounds of each quadrant'''

    min_lon, max_lon, min_lat, max_lat = quadrant
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    '''calculates center point'''

    return Point(center_lon, center_lat)

def calculate_center_points(quadrants):

    '''iterates through all quadrants 
    and saves center points into a list'''

    center_points = []
    for quadrant in quadrants:
        center_point = calculate_quadrant_center(quadrant)
        center_points.append(center_point)
    
    '''output is put into given API'''

    return center_points


def get_bounding_box_from_geometry(geo: Union[Polygon, MultiPolygon]):
        """Converts a Polygon or MultiPolygon to a bounding box."""
        if isinstance(geo, Polygon):
            min_lon, min_lat, max_lon, max_lat = geo.bounds
        elif isinstance(geo, MultiPolygon):
            min_lon, min_lat, max_lon, max_lat = geo.bounds  # MultiPolygon bounds gives overall bounding box
        else:
            raise ValueError("geo must be a Polygon or MultiPolygon")
        return min_lon, min_lat, max_lon, max_lat