"""
"""

# Standard library imports
import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Iterable, List


class IRoutingClient(ABC):
    """ """

    pass


@dataclass
class Range:
    """"""

    min: int
    max: int
    step: int

    @property
    def values(self) -> List[int]:
        """ """
        return list(range(self.min, self.max + self.step, self.step))


@dataclass
class SimulationParams:
    """Single simulation."""

    place_capacity_column: int
    num_vehicles: int
    vehicle_capacity: int
    runtime: int


@dataclass
class ParameterSweep:
    """"""

    name: str
    place_capacity_column: str
    num_vehicles: Range
    vehicle_capacity: Range
    simulation_runtime: Range

    @classmethod
    def from_config(cls, config: Dict):
        """ """
        try:
            return ParameterSweep(
                name=config["name"],
                place_capacity_column=config["place_capacity_column"],
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
        """ """
        simulation_runtimes = self.simulation_runtime.values
        vehicle_counts = self.num_vehicles.values
        vehicle_capacities = self.vehicle_capacity.values
        for runtime, vehicle_count, vehicle_capacity in itertools.product(
            simulation_runtimes, vehicle_counts, vehicle_capacities
        ):
            yield (
                SimulationParams(
                    self.place_capacity_column, vehicle_count, vehicle_capacity, runtime
                )
            )
