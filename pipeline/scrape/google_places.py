"""Provides access to geographic locations using the Google Places API.
"""

# Standard library imports
import logging
import os
from typing import Dict, List, Union

# Third-party imports
import requests
from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd

# Application imports
from pipeline.scrape.common import IPlacesProvider


class GooglePlacesClient(IPlacesProvider):
    """A simple wrapper for the Google Places API.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `GooglePlacesClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if an environment variable, 
                `GOOGLE_MAPS_API_KEY`, is not found.
        
        Returns:
            `None`
        """
        try:
            self._api_key = os.environ["GOOGLE_MAPS_API_KEY"]
            self._logger = logger
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize GooglePlacesClient."
                f"Missing expected environment variable \"{e}\"."
            ) from None
        
        
    def find_places_in_geography(
        self, 
        geojson_file_path: str
    ) -> List[Dict]:
        """Locates all POIs within the given geography.

        Documentation: # TODO: Cite whatever resources you use here:
        - ["Overview | Places API"](https://developers.google.com/maps/documentation/places/web-service/overview)

        Args:
            geojson_file_path (str): Path to the GeoJSON file.

        Returns:
            (`list` of `dict`): The list of places.
        """
        # Read the GeoJSON file using geopandas
        gdf = gpd.read_file("/Users/lydialo/Data Clinic Perpetual/2024-winter-perpetual/data/boundaries/hilo.geojson")

        # Extract the combined geometry from the GeoDataFrame
        combined_geometry = gdf.unary_union

        # Define the Places API endpoint
        places_endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

        # Define search parameters based on the provided geometry
        params = {
            "key": self._api_key,
            "radius": 500,  # Adjust the radius as needed
            "type": "restaurant",  # You can customize the type of places you are looking for
            "keyword": "business",  # You can customize the keyword
        }

        # Extract the center point of the geometry for API request
        if combined_geometry.geom_type == Polygon:
            center_point = combined_geometry.representative_point().coords[0]
        elif combined_geometry.geom_type == MultiPolygon:
            center_point = combined_geometry.centroid.coords[0]
        else:
            raise ValueError("Unsupported geometry type")

        params["location"] = f"{center_point[1]},{center_point[0]}"

        # Make the API request
        response = requests.get(places_endpoint, params=params)
        results = response.json().get("results", [])

        # Process the API response and extract relevant information
        places_list = []
        for result in results:
            place_info = {
                "name": result.get("name"),
                "address": result.get("vicinity"),
                "latitude": result["geometry"]["location"]["lat"],
                "longitude": result["geometry"]["location"]["lng"],
            }
            places_list.append(place_info)

        return places_list

