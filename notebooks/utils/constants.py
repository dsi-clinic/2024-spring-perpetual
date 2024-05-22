"""Constants used across notebooks.
"""

# Standard library imports
import pathlib

# Directories
BASE_DIR = pathlib.Path(__file__).parents[2]
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DATA_DIR = BASE_DIR / "data"
BOUNDARIES_DIR = DATA_DIR / "boundaries"
FOOT_TRAFFIC_DIR = DATA_DIR / "foot-traffic"
INFOGROUP_DIR = DATA_DIR / "infogroup"
TRIPADVISOR_DIR = DATA_DIR / "tripadvisor"

# File paths
INFOGROUP_2023_FPATH = INFOGROUP_DIR / "2023_Business_Academic_QCQ.txt"
