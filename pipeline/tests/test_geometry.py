"""Unit tests for structures and functions in the geometry utility module.
"""

# Standard library imports
import math

# Third-party imports
import pytest
from shapely import box

# Appliction imports
from pipeline.utils.geometry import BoundingBox


@pytest.mark.parametrize(
    "test_name,min_x,max_x,min_y,max_y,x_split,y_split,expected_width,expected_height",
    [
        ("Unit Square - 10 Slices", 0.0, 1.0, 0.0, 1.0, 10, 1, 0.1, 1.0),
        ("Shifted Unit Square - 10 Slices", -1.0, 0.0, -1.0, 0.0, 10, 1, 0.1, 1.0),
    ],
)
def test_create_bbox_subdivisions_rectangles_success(
    test_name: str,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    x_split: int,
    y_split: int,
    expected_width: float,
    expected_height: float,
):
    """Confirms that a bounding box can be split along its axes."""
    bbox = BoundingBox(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    bboxes = bbox.split_along_axes(x_into=x_split, y_into=y_split)
    assert len(bboxes) == (x_split * y_split)
    assert all(math.isclose(b.width, expected_width) for b in bboxes)
    assert all(math.isclose(b.height, expected_height) for b in bboxes)
