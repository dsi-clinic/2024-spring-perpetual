"""Retrieves locations from one or more providers (e.g., Google Places, Yelp).
"""

# Third-party imports
from django.contrib.gis.geos import Point
from django.core.management.base import BaseCommand, CommandParser
from shapely import wkt

# Application imports
from common.logger import LoggerFactory
from foodware.models import (
    FoodwareProject,
    FoodwareProjectBin,
    PoiCache,
    PoiParentCategory,
    PoiProvider,
    PoiProviderCategory,
)
from foodware.places import IPlacesProvider, IPlacesProviderFactory


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
            choices=["bing", "google", "tomtom", "tripadvisor", "yelp"],
            default=["google", "tripadvisor"],
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

        # Fetch POI within each boundary and category using given provider(s)
        cached_results = []
        for category in (
            PoiParentCategory.Name.RESTAURANTS,
            PoiParentCategory.Name.LODGING,
        ):
            # Log start of process
            self._logger.info(
                f'Received request to fetch points of interest related to "{category}".'
            )

            # Get parent POI category by name from database
            self._logger.info(f'Fetching "{category}" parent category from database.')
            parent_cat = PoiParentCategory.objects.get(name=category)

            # Fetch POI from each provider
            for provider_name in options["providers"]:

                # Get provider by name from database
                self._logger.info(
                    f'Fetching requested POI provider "{provider_name}" from database.'
                )
                provider = PoiProvider.objects.get(name=provider_name)

                # Attempt to fetch data from cache
                # try:
                #     self._logger.info("Attempting to load POI from cache.")
                #     cached = PoiCache.objects.get(
                #         project=project,
                #         provider=provider,
                #         parent_category=parent_cat,
                #     )
                #     cached_results.append(cached)
                #     continue
                # except PoiCache.DoesNotExist:
                #     self._logger.info(
                #         "Cached data not found for project, provider, and category."
                #     )
                #     pass

                # If no data found, get provider POI categories under parent category
                self._logger.info(
                    f"Querying database for active POI categories related "
                    f'to "{category}" from provider "{provider_name}".'
                )
                provider_categories = list(
                    PoiProviderCategory.objects.filter(
                        parent=parent_cat, provider=provider, active=True
                    )
                    .values_list("name", flat=True)
                    .all()
                )
                self._logger.info(f"{len(provider_categories)} record(s) found.")

                # Skip provider if it cannot service POI category request
                if not provider_categories:
                    self._logger.info(
                        f'Provider "{provider_name}" cannot fetch '
                        "this type of POI category. Skipping processing."
                    )
                    continue

                # Perform nearby search for POIs
                self._logger.info(
                    f'Searching for "{category}" POI in the locale of '
                    f'"{project.locale.name}" using {provider_name}.'
                )
                client: IPlacesProvider = IPlacesProviderFactory.create(
                    provider_name, self._logger
                )
                result = client.run_nearby_search(polygon, provider_categories)
                self._logger.info(
                    f"Found {len(result.clean)} POIs with {len(result.errors)} errors."
                )

                # Cache results
                self._logger.info("Caching results in database.")
                cached, _ = PoiCache.objects.update_or_create(
                    project=project,
                    provider=provider,
                    parent_category=parent_cat,
                    defaults={
                        "data": {
                            "clean": result.clean,
                            "raw": result.raw,
                            "errors": result.errors,
                        }
                    },
                )

                # Append results to list
                cached_results.append(cached)

        # Map POI data to bins
        self._logger.info("Mapping collected POI to project bins.")
        bins = []
        for cached in cached_results:
            for poi in cached.data["clean"]:
                bins.append(
                    FoodwareProjectBin(
                        project=cached.project,
                        provider=cached.provider,
                        parent_category=cached.parent_category,
                        external_id=poi["id"],
                        classification=FoodwareProjectBin.Classification.UNDETERMINED,
                        name=poi["name"],
                        external_categories=poi["aliases"],
                        formatted_address=poi["address"],
                        coords=Point(float(poi["lon"]), float(poi["lat"])),
                        notes=poi["notes"] or "",
                        features=poi["features"],
                    )
                )

        # Bulk insert records into database
        self._logger.info(f"Bulk inserting {len(bins)} record(s) into database.")
        created = FoodwareProjectBin.objects.bulk_create(
            bins, batch_size=1000, ignore_conflicts=True
        )
        self._logger.info(
            f"{len(created)} record(s) successfully inserted or ignored on conflict."
        )

        # Log successful completion and end process
        self._logger.info("Process completed successfully.")
