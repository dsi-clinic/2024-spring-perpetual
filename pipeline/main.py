"""The entrypoint for the pipeline application.
"""

# Standard library imports
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Union

# Third-party imports
import pandas as pd
import yaml

# Application imports
from pipeline.constants import CITY_BOUNDARIES_DIR, OUTPUT_DIR, PIPELINE_DIR, POI_DIR
from pipeline.routes.common import ParameterSweep
from pipeline.routes.google_or import GoogleORToolsClient
from pipeline.routes.visualize import route_to_plain_text, visualize_routes
from pipeline.scrape import IPlacesProviderFactory
from pipeline.utils.logger import LoggerFactory, logging
from pipeline.utils.storage import IDataStore, IDataStoreFactory
from shapely import MultiPolygon, Polygon
from shapely.geometry import shape


def fetch_poi(
    city: str,
    polygon: Union[MultiPolygon, Polygon],
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


def compute_routes(
    city: str,
    boundary: Union[MultiPolygon, Polygon],
    solver: str,
    locations_df: pd.DataFrame,
    distances_df: pd.DataFrame,
    pickups_sweep: ParameterSweep,
    combo_sweep: ParameterSweep,
    route_colors: List[str],
    storage: IDataStore,
    logger: logging.Logger,
):
    """ """
    # Initialize route optimization client
    client = GoogleORToolsClient()

    # Define output folder to store results of parameter sweep experiment
    exp_name = "_".join(combo_sweep.name.lower().split())
    exp_dir = f"{OUTPUT_DIR}/{exp_name}"

    # Iterate through combinations of parameters for "pickups-only" simulations
    for pickup_params in pickups_sweep.yield_simulation_params():

        # Iterate through combinations of parameters for
        # combined "pickup and dropoff" simulations
        for combo_params in combo_sweep.yield_simulation_params():

            # Mark start of simulation
            sim_started_utc = datetime.utcnow()

            # Create folder name for simulation assets
            sim_dir = f"{exp_dir}/{sim_started_utc.strftime('%Y%m%d%H%M%S')}"

            # Execute with selected parameters
            logger.info("Executing next simulation.")
            routes_df = client.solve_bidirectional_cvrp(
                locations_df, distances_df, pickup_params, combo_params
            )
            num_routes = 0 if routes_df is None else len(routes_df["Route"].unique())
            logger.info(f"Simulation complete. {num_routes} route(s) found.")

            # Define simulation metadata
            metadata = {
                "Name": combo_sweep.name,
                "Type": "Multi-Stage Simulation",
                "Start Time (UTC)": sim_started_utc.isoformat(),
                "End Time (UTC)": datetime.utcnow().isoformat(),
                "Solver": solver,
                "City": city,
                "Num Routes Generated": num_routes,
                "Pickup Only Parameters": {
                    "Number of Vehicles": num_routes,
                    "Vehicle Capacity": pickup_params.vehicle_capacity,
                    "Simulation Runtime (Seconds)": pickup_params.runtime,
                },
                "Combined Pickup-Dropoff Parameters": {
                    "Number of Vehicles": combo_params.num_vehicles,
                    "Vehicle Capacity": combo_params.vehicle_capacity,
                    "Simulation Runtime (Seconds)": combo_params.runtime,
                },
            }

            # Write metadata to JSON file
            logger.info("Writing simulation metadata to JSON file.")
            with storage.open_file(f"{sim_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # If no solution was found, skip to next set of parameters
            if routes_df is None:
                continue

            # Otherwise, write routes to file
            logger.info("Writing route solutions to CSV file.")
            with storage.open_file(f"{sim_dir}/routes.csv", "w") as f:
                routes_df.to_csv(f, index=False)

            # Visualize each route using plaintext
            route_strs = []
            grouped = routes_df.groupby("Route")
            for key in grouped.groups:
                grp_df = grouped.get_group(key)
                route_strs.append(route_to_plain_text(key, grp_df))

            # Write result to text file
            with storage.open_file(f"{sim_dir}/routes.txt", "w") as f:
                f.write("\n\n".join(route_strs))

            # Visualize routes as map
            maps = visualize_routes(routes_df, boundary, route_colors)

            # Write results to HTML file
            for name, map_obj in maps:
                with storage.open_file(f"{sim_dir}/{name}.html", "w") as f:
                    map_str = map_obj.get_root().render()
                    f.write(map_str)


def main(
    args: argparse.Namespace,
    config: Dict,
    storage: IDataStore,
    logger: logging.Logger,
) -> None:
    """Fetches points of interest to use as indoor and outdoor
    points of collection and distribution.

    Args:
        logger (`logging.Logger`): An instance of a standard logger.

    Returns:
        `None`
    """
    # Load geography from file
    logger.info("Loading geography from GeoJSON file.")
    city_fpath = f"{CITY_BOUNDARIES_DIR}/{args.city}.geojson"
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

    # Fetch points of interest within extracted boundary
    logger.info(f'Fetching points of interest for city "{args.city}".')
    places_stage = config["stages"]["places"]
    places = fetch_poi(
        args.city,
        polygon,
        args.providers,
        places_stage["use_cached"],
        storage,
        logger,
    )
    logger.info(
        f"{len(places)} place(s) successfully retrieved across "
        f"{len(args.providers)} third-party geolocation provider(s)."
    )

    # Classify points of interest as indoor and outdoor bin locations
    with storage.open_file("galveston_inputs/indoor_outdoor_pts_galv.csv", "r") as f:
        locations_df = pd.read_csv(f)
    with storage.open_file(
        "galveston_inputs/indoor_outdoor_distances_galv.csv", "r"
    ) as f:
        distances_df = pd.read_csv(f)

    # Create distance matrix for every unique pair of locations

    # Run route simulation for locations and visualize results
    routes_stage = config["stages"]["routes"]
    sim_config = routes_stage["experiments"]
    viz_config = routes_stage["visualizations"]
    pickups_sweep = ParameterSweep.from_config(sim_config["pickups_only"])
    combo_sweep = ParameterSweep.from_config(sim_config["pickups_and_dropoffs"])
    compute_routes(
        args.city,
        polygon,
        args.solver,
        locations_df,
        distances_df,
        pickups_sweep,
        combo_sweep,
        viz_config["colors"],
        storage,
        logger,
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
            "-p",
            "--providers",
            nargs="+",
            choices=["bing", "google", "tomtom", "yelp"],
        )
        parser.add_argument(
            "-s", "--solver", choices=["google", "gurobi"], default="google"
        )
        args = parser.parse_args()

        # Determine current development environment
        try:
            env = os.environ["ENV"]
        except KeyError as e:
            raise RuntimeError(
                f'Missing required environment variable "{e}".'
            ) from None

        # Read and parse local configuration file
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
