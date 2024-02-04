"""Unit tests for structures and functions in the geometry utility module.
"""

# Standard library imports
import math

# Third-party imports
import pytest
# Appliction imports
from pipeline.utils.geometry import create_bbox_subdivisions
from shapely import box


@pytest.mark.parametrize(
    "test_name,min_x,max_x,min_y,max_y,num_bbox,expected_width,expected_height",
    [
        ("Unit Square - 10 Slices", 0.0, 1.0, 0.0, 1.0, 10, 0.1, 1.0),
        ("Shifted Unit Square - 10 Slices", -1.0, 0.0, -1.0, 0.0, 10, 0.1, 1.0),
    ],
)
def test_create_bbox_subdivisions_rectangles_success(
    test_name: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    num_bbox: int,
    expected_width: float,
    expected_height: float,
):
    """Confirms that `create_bbox_subdivisions` creates the
    expected number and shape of bounding boxes given valid
    coordinates for a rectangle.
    """
    polygon = box(xmin=min_x, ymin=min_y, xmax=max_x, ymax=max_y)
    bboxes = create_bbox_subdivisions(polygon, num_bbox)
    assert len(bboxes) == num_bbox
    assert all(math.isclose(b.width, expected_width) for b in bboxes)
    assert all(math.isclose(b.height, expected_height) for b in bboxes)
