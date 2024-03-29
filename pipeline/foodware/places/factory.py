"""Factories used throughout the package.
"""

# Application imports
from foodware.places.bing import BingMapsClient
from foodware.places.common import IPlacesProvider
from foodware.places.google_places import GooglePlacesClient
from foodware.places.tomtom import TomTomSearchClient
from foodware.places.yelp import YelpClient
from common.logger import logging


class IPlacesProviderFactory:
    """A simple factory for instantiating an `IPlaceProvider`."""

    _REGISTRY = {
        "bing": BingMapsClient,
        "google": GooglePlacesClient,
        "tomtom": TomTomSearchClient,
        "yelp": YelpClient,
    }

    @staticmethod
    def create(name: str, logger: logging.Logger) -> IPlacesProvider:
        """Instantiates an `IPlacesProvider` by name.

        Args:
            name (`str`): The name.

            logger (`logging.Logger`): A standard logger instance.

        Returns:
            (`IPlacesProvider`): The provider.
        """
        try:
            provider = IPlacesProviderFactory._REGISTRY[name]
            return provider(logger)
        except KeyError as e:
            raise RuntimeError(
                "Requested a points of interest provider that "
                f'has not been registered, "{e}". Expected one of '
                ", ".join(f'"{k}"' for k in IPlacesProviderFactory._REGISTRY.keys())
            ) from None
