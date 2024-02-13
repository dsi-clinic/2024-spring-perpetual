"""The entrypoint for the pipeline application.
"""

# Standard library imports
import json
from typing import Dict, List

# Third-party imports
from shapely.geometry import shape

# Application imports
from pipeline.scrape import (
    IPlacesProvider,
    BingMapsClient,
    GooglePlacesClient,
    TomTomSearchClient,
    YelpClient
)
from pipeline.utils.logger import logging, LoggerFactory
from pipeline.utils.storage import IDataReaderFactory


def main(config: Dict, logger: logging.Logger) -> None:
    """Fetches points of interest to use as indoor and outdoor
    points of collection and distribution.

    Args:
        logger (`logging.Logger`): An instance of a standard logger.

    Returns:
        `None`
    """
    # Initialize storage clients
    logger.info("Initializing storage client.")
    storage = IDataReaderFactory.get()

    # Load geography from file
    # TODO: Use configuration for file paths
    logger.info("Loading geography from file.")
    with storage.read_file("boundaries/hilo.geojson") as f:
        data = json.load(f)
        
    # Parse geography for GeoJSON and convert to Shapely object,
    # assumed to be a Polygon or Multipolygon
        
    # Commented out for Galveston
    # geojson = data["features"][0]["geometry"]
    geojson = data
    polygon = shape(geojson)

     # Define the type of places you're interested in
    place_type = 'restaurant'  # for example

    # Call clients and aggregate results
    places = []
    locators = [
        # BingMapsClient,
        GooglePlacesClient,
        # TomTomSearchClient,
        YelpClient
    ]
    for locator_cls in locators:
        locator: IPlacesProvider = locator_cls(logger)
        api_places = locator.find_places_in_geography(polygon, place_type) or []
        places.extend(api_places)

    # Write results to file
    # TODO - Implement storage writer
    with open("data/output_poi.json", "w") as f:
        json.dump(places, f, indent=2)



if __name__ == "__main__":
    try:
        # TODO - Could read in configuration files or 
        # parse command line arguments here if need be
        # Read in YAML file and parse as dictionary

        config = {}
        logger = LoggerFactory.get("PIPELINE")
        main(config, logger)
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution. {e}")
        exit(1)
