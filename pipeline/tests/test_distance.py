"""Unit tests for distance calculations between coordinates.
"""

# Standard library imports
from typing import List

# Third-party imports
import numpy as np
import pytest

# Appliction imports
from pipeline.distance.mapbox import (
    MapboxApiClient,
    MapboxRouteEnum,
    MapboxRoutingProfile,
)
from pipeline.utils.geometry import Coordinate


@pytest.fixture
def coords():
    """Coordinates to use for testing."""
    return [
        Coordinate(lat=37.78, lon=-122.42),
        Coordinate(lat=37.91, lon=-122.45),
        Coordinate(lat=37.73, lon=-122.48),
    ]


def test_mapbox_make_distance_matrix(coords: List[Coordinate]) -> None:
    """Confirms that `make_distance_matrix` does not raise an exception with standard input."""
    client = MapboxApiClient()
    profile = MapboxRoutingProfile(name=MapboxRouteEnum.DRIVING)
    matrix = client.make_distance_matrix(coords, profile)
    # assert matrix dimensionality and data type
