"""Constants used across notebooks.
"""

# Standard library imports
import pathlib

BASE_DIR = pathlib.Path(__file__).parents[2]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DATA_DIR = BASE_DIR / "data"

ADDRESS_MAPPING = {
    "STREET": "ST",
    "ROAD": "RD",
    "AVENUE": "AVE",
    "BOULEVARD": "BLVD",
    "LANE": "LN",
    "DRIVE": "DR",
    "COURT": "CT",
    "NORTH": "N",
    "SOUTH": "S",
    "EAST": "E",
    "WEST": "W",
    "PLACE": "PL",
    "SQUARE": "SQ",
    "UNIT": "UNIT",
    "APARTMENT": "APT",
    "SUITE": "STE",
    "FLOOR": "FL",
    "BUILDING": "BLDG"
}

HOTEL_LR_DEPENDENT_VARIABLES = ["parent_sales_volume", "sales_volume", "employee_size"]
HOTEL_LR_INDEPENDENT_VARIABLES = ["large_hotel", "number_of_rooms", "num_reviews", "price_level_category", "rating"]
HOTEL_LR_VARIABLES = HOTEL_LR_DEPENDENT_VARIABLES + HOTEL_LR_INDEPENDENT_VARIABLES