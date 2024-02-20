"""The entrypoint for the pipeline application.
"""

# Standard library imports
import argparse
import json
import os
from typing import Dict, List

# Third-party imports
import yaml
from shapely.geometry import shape

# Application imports
from pipeline.constants import CITY_BOUNDARIES_DIR, PIPELINE_DIR, POI_DIR
from pipeline.scrape import IPlacesProviderFactory
from pipeline.utils.logger import logging, LoggerFactory
from pipeline.utils.storage import IDataStore, IDataStoreFactory


def fetch_poi(
    city: str,
    providers: List[str],
    use_cached: bool,
    storage: IDataStore,
    logger: logging.Logger,
) -> List[Dict]:
    """Fetches points of interest for a given city for use as
    indoor and outdoor points of foodware collection and distribution.

    Args:
        city (`str`): The name of the city (e.g., "hilo" or "galveston").

        providers (`list` of `str`): The names of the points of
            interest API providers to query (e.g., "bing" for Microsoft
            Bing Maps, "google" for Google Maps, "tomtom" for TomTom,
            or "yelp" for Yelp Fusion).

        use_cached (`bool`): A boolean indicating whether points of interest
            previously fetched and cached from a provider for development
            purposes should be returned, rather than making a new API call.

        storage (`IDataStore`): A client for reading and writing to
            a local or cloud-based data store.

        logger (`logging.Logger`): An instance of a standard logger.

    Returns:
        `None`
    """
    # Load geography from file
    logger.info("Loading geography from GeoJSON file.")
    city_fpath = f"{CITY_BOUNDARIES_DIR}/{city}.geojson"
    try:
        with storage.open_file(city_fpath) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(
            f"POI fetch failed. Could not locate input boundary "
            f'file "{city_fpath}" in the configured data directory.'
        ) from None
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f'POI fetch failed. Input boundary file "{city_fpath}" '
            f"contains invalid JSON that cannot be processed. {e}"
        ) from None

    # Extract GeoJSON geometry and convert to Shapely object
    try:
        geometry = data["features"][0]["geometry"]
        polygon = shape(geometry)
    except (KeyError, IndexError, AttributeError):
        raise RuntimeError(
            "POI fetch failed. The input boundary is not valid GeoJSON."
        ) from None

    # Call third-party geolocation clients while caching and aggregating results
    places = []
    for provider in providers:
        # Define relative path to output POI file within data directory
        provider_poi_fpath = f"{POI_DIR}/{city}_{provider}.json"
        provider_err_fpath = f"{POI_DIR}/{city}_{provider}_errors.json"

        # Attempt to load places from cache, if specified
        if use_cached:
            try:
                logger.info("Attempting to load places from cached file.")
                with storage.open_file(provider_poi_fpath, "r") as f:
                    provider_places = json.load(f)
                logger.info(f"{len(provider_places)} place(s) from {provider} found.")
                places.extend(provider_places)
                continue
            except FileNotFoundError:
                logger.info("File not found for city and provider.")
                pass

        # Find places using provider
        logger.info(f"Requesting POI data from {provider}.")
        client = IPlacesProviderFactory.create(provider, logger)
        provider_places, provider_errors = client.find_places_in_geography(polygon)
        logger.info(
            f"{len(provider_places)} place(s) from {provider} "
            f"found and {len(provider_errors)} error(s) encountered."
        )

        # Cache results to file, if any exist
        if provider_places:
            logger.info("Write provider POIs to file.")
            with storage.open_file(provider_poi_fpath, "w") as f:
                json.dump(provider_places, f, indent=2)

        # Persist errors for later inspection, if any exist
        if provider_errors:
            logger.info("Writing provider HTTP request errors to file.")
            with storage.open_file(provider_err_fpath, "w") as f:
                json.dump(provider_errors, f, indent=2)

        # Add places to aggregated list
        places.extend(provider_places)

    return places


def main(
    args: argparse.Namespace, config: Dict, storage: IDataStore, logger: logging.Logger
) -> None:
    """Fetches points of interest to use as indoor and outdoor
    points of collection and distribution.

    Args:
        logger (`logging.Logger`): An instance of a standard logger.

    Returns:
        `None`
    """
    # Extract configuration for pipeline stages
    stages = config["stages"]
    places_stage = stages["places"]

    # Fetch points of interest
    logger.info(f'Fetching points of interest for city "{args.city}".')
    places = fetch_poi(
        args.city,
        args.providers,
        places_stage["use_cached"],
        storage,
        logger,
    )
    logger.info(
        f"{len(places)} place(s) successfully retrieved across "
        f"{len(args.providers)} third-party geolocation provider(s)."
    )


if __name__ == "__main__":
    try:
        # Initialize logger
        logger = LoggerFactory.get("PIPELINE")
        logger.info("Starting pipeline execution.")

        # Initialize storage client
        logger.info("Initializing storage client.")
        storage = IDataStoreFactory.get()

        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "city", choices=["ann_arbor", "galveston", "hilo", "savannah"]
        )
        parser.add_argument(
            "-p", "--providers", nargs="+", choices=["bing", "google", "tomtom", "yelp"]
        )
        args = parser.parse_args()

        # Determine current development environment
        try:
            env = os.environ["ENV"]
        except KeyError as e:
            raise RuntimeError(
                f'Missing required environment variable "{e}".'
            ) from None

        # Read and parse local YAML configuration file
        try:
            logger.info(f'Loading configuration file for environment "{env}".')
            with open(f"{PIPELINE_DIR}/config.{env.lower()}.yaml") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError("Cannot find configuration file.")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse file contents. {e}")

        # Begin pipeline
        main(args, config, storage, logger)

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution. {e}")
        exit(1)
