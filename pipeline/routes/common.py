"""Defines interfaces and common classes for computing and analyzing routes.
"""

# Standard library imports
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List

# Third-party imports
import pandas as pd

# Appliation imports
from pipeline.utils.logger import logging


@dataclass
class Range:
    """Represents an inclusive integer range."""

    min: int
    """The inclusive minimum value/lower bound.
    """

    max: int
    """The inclusive maximum value/upper bound.
    """

    step: int
    """The steps to take between the lower and upper boundaries.
    """

    @property
    def values(self) -> List[int]:
        """Returns a list of all integer values within the range."""
        return list(range(self.min, self.max + self.step, self.step))


@dataclass
class SimulationParams:
    """Represents parameters used in a single simulation run."""

    demand_column: str
    """The name of the data column holding pickup and dropoff demands for locations.
    """

    num_vehicles: int
    """The maximum number of vehicles to use in the simulation, if possible to set.
    """

    vehicle_capacity: int
    """The maximum number of items that a vehicle can carry at once.
    """

    runtime: int
    """The maximum number of seconds the simulation should run while seeking an optimum.
    """


@dataclass
class ParameterSweep:
    """A set of parameters to test in an experiment with one or more simulation runs."""

    name: str
    """The name of the experiment (e.g., "combined_dropoffs_and_pickups").
    """

    demand_column: str
    """The name of the data column holding pickup and dropoff demands for locations.
    """

    num_vehicles: Range
    """The range of vehicle numbers to test.
    """

    vehicle_capacity: Range
    """The range of vehicle capacities to test.
    """

    simulation_runtime: Range
    """The range of simulation runtimes (in seconds) to test.
    """

    @classmethod
    def from_config(cls, config: Dict) -> "ParameterSweep":
        """Creates and initializes a new instance of a
        `ParameterSweep` from a configuration object.

        Args:
            config (`dict`): The configuration.

        Returns:
            (`ParameterSweep`): The instance.
        """
        try:
            return ParameterSweep(
                name=config["name"],
                demand_column=config["place_capacity_column"],
                num_vehicles=Range(**config["num_vehicles"]),
                vehicle_capacity=Range(**config["vehicle_capacity"]),
                simulation_runtime=Range(**config["simulation_runtime"]),
            )
        except KeyError as e:
            raise ValueError(
                "Failed to instantiate new SimulationConfig object. "
                f'Configuration argument missing expected key "{e}".'
            ) from None

    def yield_simulation_params(self) -> Iterable[SimulationParams]:
        """Calculates all unique combinations of the simulation
        parameters and then yields each combination to the caller
        one at a time through a generator.

        Args:
            `None`

        Yields:
            (`SimulationParams`): The simulation parameters.
        """
        simulation_runtimes = self.simulation_runtime.values
        vehicle_counts = self.num_vehicles.values
        vehicle_capacities = self.vehicle_capacity.values
        for runtime, vehicle_count, vehicle_capacity in itertools.product(
            simulation_runtimes, vehicle_counts, vehicle_capacities
        ):
            yield (
                SimulationParams(
                    self.demand_column, vehicle_count, vehicle_capacity, runtime
                )
            )


class IRoutingClient(ABC):
    """An abstract class for solving routing optimization problems."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of an `IRoutingClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Returns:
            `None`
        """
        self._logger = logger

    @abstractmethod
    def solve_bidirectional_cvrp(
        self,
        locations_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        pickup_params: SimulationParams,
        combo_params: SimulationParams,
    ) -> pd.DataFrame:
        """Solves a subcase of the Capacitated Vehicle Routing Problem (CVRP)
        in which trucks can pick up and drop off items at the same locations.

        Args:
            locations_df (`pd.DataFrame`): The locations to use for routing.

            distance_df (`pd.DataFrame`): A matrix of distances computed between
                every unique location in `locations_df`. Units are expressed in meters.

            pickup_params (`SimulationParams`): Configuration for the pickup route
                simulation.

            combo_params (`SimulationParams`): Configuration for the combined
                pickup/drop off route simulation.

        Returns:
            (`pd.DataFrame`): A DataFrame containing the optimal routes.
        """
        raise NotImplementedError()
