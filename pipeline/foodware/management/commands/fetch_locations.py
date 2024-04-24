"""Retrieves locations from one or more providers (e.g., Google Places, Yelp).
"""

# Standard library imports
import json
import os

# Third-party imports
from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from shapely import wkt

# Application imports
from common.logger import LoggerFactory
from common.storage import IDataStoreFactory
from foodware.classification.label import filter
from foodware.mapping.padlet import PadletClient
from foodware.models import FoodwareProject
from foodware.places import IPlacesProviderFactory


class Command(BaseCommand):
    """Fetches, standardizes, and combines points of interest (POI) data from one
    or more providers given by command line arguments. Saves the raw result
    for each provider as well as the combined and standardized result to the data
    store specified in configuration settings.

    References:
    - https://docs.djangoproject.com/en/4.1/howto/custom-management-commands/
    - https://docs.djangoproject.com/en/4.1/topics/settings/
    """

    help = "Fetches, standardizes, and combines location data across multiple sources."
    name = "Fetch Location Data"

    def __init__(self, *args, **kwargs) -> None:
        """Initializes a new instance of the `Command`.

        Args:
            *The default positional arguments for the base class.

        Kwargs:
            **The default keyword arguments for the base class.

        Returns:
            `None`
        """
        self._logger = LoggerFactory.get(Command.name.upper())
        self._storage = IDataStoreFactory.get()
        super().__init__(*args, **kwargs)

    def add_arguments(self, parser: CommandParser) -> None:
        """Requires the argument `project`, which refers to a unique identifier
        for the foodware project. Also provides an option, `--providers` (`-p`),
        to fetch location data from one or more selected sources. Valid
        choices include:

        - bing
        - google
        - tomtom
        - yelp

        Defaults to `["yelp"]` if no option is provided. Finally, a "--cached"
        option signifies that cached locations should be loaded and used in
        the cleaning step if they exist, as opposed to requesting new data from APIs.

        Args:
            parser (`CommandParser`)

        Returns:
            `None`
        """
        parser.add_argument("project")
        parser.add_argument(
            "-p",
            "--providers",
            nargs="+",
            choices=["bing", "google", "tomtom", "yelp"],
            default=["yelp"],
        )
        parser.add_argument("-c", "--cached", action="store_true")

    def handle(self, *args, **options) -> None:
        """Executes the command. If the `providers` option
        has been provided, only a subset of APIs and scrapers
        are called.

        Args:
            `None`

        Returns:
            `None`
        """
        # Fetch project corresponding to id
        try:
            project_id = options["project"]
            self._logger.info(f'Fetching foodware project "{project_id}".')
            project = FoodwareProject.objects.get(id=project_id)
        except FoodwareProject.DoesNotExist:
            self._logger.error(f'No model found with id "{project_id}".')
            exit(1)

        # Parse project's geographic boundary into Shapely object
        try:
            self._logger.info("Parsing model geography boundary.")
            polygon = wkt.loads(project.locale.geometry.wkt)
        except AttributeError as e:
            self._logger.error(
                f'Unable to convert model boundary to Shapely object. "{e}".'
            )
            exit(1)

        # Initialize list of places
        places = []

        # Iterate through each provider
        for provider in options["providers"]:

            # Define relative paths to output POI files within data directory
            model_fpath = f"{settings.POI_DIR}/{project_id}"
            provider_poi_clean_fpath = f"{model_fpath}/{provider}_clean.json"
            provider_poi_raw_fpath = f"{model_fpath}/{provider}_raw.json"
            provider_err_fpath = f"{model_fpath}/{provider}_errors.json"

            # Attempt to load places from cache, if specified
            if options["cached"]:
                try:
                    self._logger.info("Attempting to load places from cached file.")
                    with self._storage.open_file(provider_poi_clean_fpath, "r") as f:
                        provider_places = json.load(f)
                    self._logger.info(
                        f"{len(provider_places)} place(s) from {provider} found."
                    )
                    places.extend(provider_places)
                    continue
                except FileNotFoundError:
                    self._logger.info("File not found for city and provider.")
                    pass

            # Find places using provider
            self._logger.info(f"Requesting POI data from {provider} within boundary.")
            client = IPlacesProviderFactory.create(provider, self._logger)
            result = client.find_places_in_geography(polygon)
            self._logger.info(
                f"{len(result.raw)} place(s) from {provider} "
                f"found, with {len(result.clean)} remaining after "
                f"cleaning. {len(result.errors)} error(s) encountered."
            )

            # Cache raw places to file, if any exist
            if result.raw:
                self._logger.info("Writing raw provider POIs to file.")
                with self._storage.open_file(provider_poi_raw_fpath, "w") as f:
                    json.dump(result.raw, f, indent=2)

            # Cache cleaned/standardized places to file, if any exist
            if result.clean:
                self._logger.info("Writing cleaned provider POIs to file.")
                with self._storage.open_file(provider_poi_clean_fpath, "w") as f:
                    json.dump(result.clean, f, indent=2)

            # Persist errors for later inspection, if any exist
            if result.errors:
                self._logger.info("Writing provider HTTP request errors to file.")
                with self._storage.open_file(provider_err_fpath, "w") as f:
                    json.dump(result.errors, f, indent=2)

            # Add places to aggregated list
            places.extend(result.clean)

        # Log result of places scrape
        self._logger.info(f"{len(places)} point(s) of interest retrieved.")

        # If no points of interest fetched, terminate process
        if not places:
            self._logger.info("Location fetch completed successfully.")
            exit(0)

        # Open crosswalk between providers' place categories and standard categories
        with self._storage.open_file(settings.CATEGORIES_CROSSWALK_JSON_FPATH) as f:
            crosswalk = json.load(f)

        # Filter records to include only indoor and/or outdoor bins
        self._logger.info("Filtering records to include only indoor or outdoor bins.")
        bins = filter(places, crosswalk)
        self._logger.info(f"{len(bins)} bin(s) successfully identified.")

        # Parse Padlet flag from environment variables
        make_padlet = os.getenv("PADLET_SHOULD_GENERATE", "false").lower() == "true"

        # If flag set to true, create new Padlet map of places
        if make_padlet:

            # Log into Padlet
            self._logger.info("Instantiating new Padlet client.")
            padlet_client = PadletClient(self._logger)

            # Add places to new Padlet map
            self._logger.info("Creating new Padlet map and adding places as markers.")
            board = padlet_client.add_board_with_posts(bins)
            url = board["data"]["attributes"]["webUrl"]["live"]
            self._logger.info(f'Successfully created new Padlet map at "{url}".')
            self._logger.info(f"{len(bins)} bin(s) mapped on Padlet platform.")

        # Log successful completion and end process
        self._logger.info("Process completed successfully.")
        exit(0)
