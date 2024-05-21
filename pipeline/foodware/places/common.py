"""Defines interfaces and common classes for scraping points of interest.
"""

# Standard library imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third-party imports
from shapely import MultiPolygon, Point, Polygon


@dataclass
class Place:
    """Represents a generic place."""

    id: str
    """The unique identifier for the place within its external data source.
    """

    name: str
    """The display name.
    """

    categories: List[str]
    """Reader-friendly categories that have been assigned
    to the place (e.g., "restaurant", "school", "theater").
    """

    aliases: List[str]
    """Category ids that have been assigned to the place 
    (e.g., "restaurant", "school", "theater").
    """

    lat: float
    """The latitude coordinate.
    """

    lon: float
    """The longitude coordinate.
    """

    address: str
    """The place's formatted street address.
    """

    is_closed: bool
    """A boolean indicating whether the place is temporarily or permanently closed.
    """

    source: str
    """The name of the data source or API providing the place.
    """

    url: Optional[str] = None
    """A link to more information about the place.
    """

    notes: Optional[str] = ""
    """Additional information to include about the place.
    """

    features: Optional[Dict] = None
    """Additional, provider-specific properties for the place.
    """


@dataclass
class PlacesSearchResult:
    """Contains the result of a points of interest (POI) search."""

    raw: List[Dict]
    """Places returned directly from an `IPlacesProvider`.
    """

    clean: List[Dict]
    """The de-duped and standardized list of places.
    """

    errors: List[Dict]
    """Any errors that occurred when fetching the data.
    """


class IPlacesProvider(ABC):
    """An abstract class for identifying points of interest (POI)."""

    @abstractmethod
    def map_place(self, place: Dict) -> Place:
        """Maps a place fetched from a data source to a standard representation.

        Args:
            place (`dict`): The place.

        Returns:
            (`Place`): The standardized place.
        """
        raise NotImplementedError

    @abstractmethod
    def run_text_search(
        self, query: str, restriction: Optional[Union[Polygon, MultiPolygon]] = None
    ) -> PlacesSearchResult:
        """Searches for one or more places using a query string
        and an optional geography to confine the search area.
        Implementation varies by provider.

        Args:
            query (`str`): The search phrase
                (e.g., "KEN'S HOUSE OF PANCAKES, 1730 KAMEHAMEHA AVE, HILO, HI").

            geo (`Polygon` or `MultiPolygon`): The search boundary. Defaults to `None`.

        Returns:
            (`PlacesSearchResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        raise NotImplementedError

    @abstractmethod
    def run_nearby_search(
        self, geo: Union[Polygon, MultiPolygon]
    ) -> PlacesSearchResult:
        """Locates all POIs within the given geography.

        Args:
            geo (`Polygon` or `MultiPolygon`): The boundary.

        Returns:
            (`PlacesSearchResult`): The result of the geography query. Contains
                a raw list of retrieved places, a list of cleaned places,
                and a list of any errors that occurred.
        """
        raise NotImplementedError

    def clean_places(
        self, places: List[Dict], geo: Union[MultiPolygon, Polygon]
    ) -> List[Dict]:
        """Cleans a list of places newly fetched from the API by removing
        duplicates, permanently closed locations, and/or those that lie outside
        the geography. NOTE: Duplicates exist because places are frequently
        tagged with multiple categories.

        Args:
            places (`list` of `dict`): The places to clean.

            geo (`shapely.MultiPolygon`|`shapely.Polygon`): The geography boundary.

        Returns:
            (`list` of `dict`): The cleaned places.
        """
        visited_ids = set()
        cleaned = []
        for place in places:

            # Map place
            mapped = self.map_place(place)

            # Filter out dupes and places outside bounds
            if (
                (mapped.id in visited_ids)
                or (mapped.is_closed)
                or (not geo.contains(Point(mapped.lon, mapped.lat)))
            ):
                continue

            # If valid place, map to standard format and append to list
            cleaned.append(vars(mapped))

            # Mark place as seen
            visited_ids.add(mapped.id)

        return cleaned
