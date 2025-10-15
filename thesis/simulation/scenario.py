"""Simulation scenario definitions and configuration management."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from thesis.common.config import (
    DATA_DIRNAME,
    FCD_CSV_SUFFIX,
    FCD_PARQUET_SUFFIX,
    NETWORK_BASE_FILENAME,
    NETWORK_RAIN_FILENAME,
    RANDOM_SEED_RAIN,
    RANDOM_SEED_TEST,
    RANDOM_SEED_TRAIN,
    SUMOCFG_SUFFIX,
    TRAFFIC_GENERATION_PERIODS,
    TRAFFIC_GENERATION_PERIODS_MEAN,
    TRAFFIC_GENERATION_PERIODS_STD,
    TRIPS_SUFFIX,
)
from thesis.common.enums import SimulationScenario


@dataclass(frozen=True, slots=True)
class SimulationScenarioConfig:
    """
    A simulation scenario config.

    Attributes:
        scenario (SimulationScenario): Scenario.
        simulation_dir (Path): Directory for the simulation experiment.
    """

    scenario: SimulationScenario
    simulation_dir: Path

    _NETWORK_PATHS: ClassVar[dict[SimulationScenario, Path]] = {
        SimulationScenario.TRAIN: NETWORK_BASE_FILENAME,
        SimulationScenario.TEST: NETWORK_BASE_FILENAME,
        SimulationScenario.RAIN: NETWORK_RAIN_FILENAME,
    }
    _RANDOM_SEEDS: ClassVar[dict[SimulationScenario, int]] = {
        SimulationScenario.TRAIN: RANDOM_SEED_TRAIN,
        SimulationScenario.TEST: RANDOM_SEED_TEST,
        SimulationScenario.RAIN: RANDOM_SEED_RAIN,
    }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.scenario=}, "
            f"{self.simulation_dir=}, "
            f"{self.data_dir=}, "
            f"{self.network_path=}, "
            f"{self.trips_path=}, "
            f"{self.fcd_csv_path=}, "
            f"{self.sumocfg_path=}, "
            f"{self.traffic_generation_periods=}, "
            f"{self.random_seed=}, "
            f"{self.fcd_parquet_path=})"
        )

    @property
    def data_dir(self) -> Path:
        """Path to the data directory."""
        return self.simulation_dir / DATA_DIRNAME

    @property
    def network_path(self) -> Path:
        """Path to the network file."""
        return self.simulation_dir / self._NETWORK_PATHS[self.scenario]

    @property
    def trips_path(self) -> Path:
        """Path to the trips file."""
        return self.simulation_dir / f"{self.scenario}{TRIPS_SUFFIX}"

    @property
    def fcd_csv_path(self) -> Path:
        """Path to the fcd CSV file."""
        return self.data_dir / f"{self.scenario}{FCD_CSV_SUFFIX}"

    @property
    def sumocfg_path(self) -> Path:
        """Path to the sumocfg file."""
        return self.simulation_dir / f"{self.scenario}{SUMOCFG_SUFFIX}"

    @property
    def traffic_generation_periods(self) -> list[float]:
        """Traffic generation periods."""
        return [
            p * self._rng.normal(TRAFFIC_GENERATION_PERIODS_MEAN, TRAFFIC_GENERATION_PERIODS_STD)
            for p in TRAFFIC_GENERATION_PERIODS
        ]

    @property
    def random_seed(self) -> int:
        """Random seed."""
        return self._RANDOM_SEEDS[self.scenario]

    @property
    def fcd_parquet_path(self) -> Path:
        """Path to the fcd Parquet file."""
        return self.data_dir / f"{self.scenario}{FCD_PARQUET_SUFFIX}"

    @property
    def _rng(self) -> np.random.Generator:
        """Random number generator."""
        return np.random.default_rng(self.random_seed)
