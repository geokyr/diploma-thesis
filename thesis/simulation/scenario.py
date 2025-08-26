"""
Simulation scenario definitions and configuration management.
Provides simulation scenario types (train/test/rain) and configuration classes for organizing different experimental conditions with appropriate network and parameter settings.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
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

logger = logging.getLogger(__name__)


class SimulationScenario(StrEnum):
    """
    Simulation Scenarios.

    Attributes:
        TRAIN: Train data on base network.
        TEST: Test data on base network.
        RAIN: Retrain/test data on rain network.
    """

    TRAIN = "train"
    TEST = "test"
    RAIN = "rain"


@dataclass(frozen=True, slots=True)
class SimulationScenarioConfig:
    """
    A simulation scenario config.

    Attributes:
        scenario (SimulationScenario): Scenario.
        simulation_dir (Path): Directory for the simulation experiment.

    Properties:
        data_dir (Path): Path to the data directory.
        network_path (Path): Path to the network file.
        trips_path (Path): Path to the trips file.
        fcd_csv_path (Path): Path to the fcd CSV file.
        sumocfg_path (Path): Path to the sumocfg file.
        traffic_generation_periods (list[float]): Traffic generation periods.
        random_seed (int): Random seed.
        fcd_parquet_path (Path): Path to the fcd Parquet file.
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
        return self.simulation_dir / DATA_DIRNAME

    @property
    def network_path(self) -> Path:
        return self.simulation_dir / self._NETWORK_PATHS[self.scenario]

    @property
    def trips_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}{TRIPS_SUFFIX}"

    @property
    def fcd_csv_path(self) -> Path:
        return self.data_dir / f"{self.scenario}{FCD_CSV_SUFFIX}"

    @property
    def sumocfg_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}{SUMOCFG_SUFFIX}"

    @property
    def traffic_generation_periods(self) -> list[float]:
        return [
            p * self._rng.normal(TRAFFIC_GENERATION_PERIODS_MEAN, TRAFFIC_GENERATION_PERIODS_STD)
            for p in TRAFFIC_GENERATION_PERIODS
        ]

    @property
    def random_seed(self) -> int:
        return self._RANDOM_SEEDS[self.scenario]

    @property
    def fcd_parquet_path(self) -> Path:
        return self.data_dir / f"{self.scenario}{FCD_PARQUET_SUFFIX}"

    @property
    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_seed)
