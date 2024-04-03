"""Factories used throughout the package.
"""

# Application imports
from pipeline.routes.common import IRoutingClient
from pipeline.routes.google_or import GoogleORToolsClient
from pipeline.routes.gurobi import GurobiClient
from pipeline.utils.logger import logging


class IRoutingClientFactory:
    """A simple factory for instantiating an `IRoutingClient`."""

    _REGISTRY = {
        "google": GoogleORToolsClient,
        "gurobi": GurobiClient,
    }

    @staticmethod
    def create(name: str, logger: logging.Logger) -> IRoutingClient:
        """Instantiates an `IRoutingClient` by name.

        Args:
            name (`str`): The name.

            logger (`logging.Logger`): A standard logger instance.

        Returns:
            (`IRoutingClient`): The client.
        """
        try:
            client = IRoutingClientFactory._REGISTRY[name]
            return client(logger)
        except KeyError as e:
            raise RuntimeError(
                "Requested a routing optimizer that "
                f'has not been registered, "{e}". Expected one of '
                ", ".join(f'"{k}"' for k in IRoutingClientFactory._REGISTRY.keys())
            ) from None
