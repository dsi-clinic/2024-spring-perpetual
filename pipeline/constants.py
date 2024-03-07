"""Constants used across the application.
"""

import pathlib

# File paths
ROOT_DIR = pathlib.Path(__file__).parents[1]
PIPELINE_DIR = ROOT_DIR / "pipeline"
DATA_DIR = ROOT_DIR / "data"
CITY_BOUNDARIES_DIR = "boundaries"
OUTPUT_DIR = "output"
POI_DIR = "poi"
