import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from thesis.common.config import (
    DATA_DIRNAME,
    EMISSION_XML_SUFFIX,
    FCD_CSV_SUFFIX,
    FCD_XML_SUFFIX,
    NETWORK_BASE_FILENAME,
    NETWORK_RAIN_FILENAME,
    RANDOM_SEED_RAIN,
    RANDOM_SEED_TEST,
    RANDOM_SEED_TRAIN,
    SUMOCFG_SUFFIX,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TRIPS_SUFFIX,
)
from thesis.common.enums import SimulationScenario

logger = logging.getLogger(__name__)


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
        emission_xml_path (Path): Path to the emission XML file.
        fcd_xml_path (Path): Path to the fcd XML file.
        sumocfg_path (Path): Path to the sumocfg file.
        traffic_generation_periods (tuple[float, ...]): Traffic generation periods.
        random_seed (int): Random seed.
        fcd_csv_path (Path): Path to the fcd CSV file.
    """

    scenario: SimulationScenario
    simulation_dir: Path

    _NETWORK_PATHS: ClassVar[dict[SimulationScenario, Path]] = {
        SimulationScenario.TRAIN: NETWORK_BASE_FILENAME,
        SimulationScenario.TEST: NETWORK_BASE_FILENAME,
        SimulationScenario.RAIN: NETWORK_RAIN_FILENAME,
    }
    _TRAFFIC_GENERATION_PERIODS: ClassVar[dict[SimulationScenario, tuple[float, ...]]] = {
        SimulationScenario.TRAIN: TRAIN_TRAFFIC_GENERATION_PERIODS,
        SimulationScenario.TEST: TEST_TRAFFIC_GENERATION_PERIODS,
        SimulationScenario.RAIN: TEST_TRAFFIC_GENERATION_PERIODS,
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
            f"{self.emission_xml_path=}, "
            f"{self.fcd_xml_path=}, "
            f"{self.sumocfg_path=}, "
            f"{self.traffic_generation_periods=}, "
            f"{self.random_seed=}, "
            f"{self.fcd_csv_path=})"
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
    def emission_xml_path(self) -> Path:
        return self.data_dir / f"{self.scenario}{EMISSION_XML_SUFFIX}"

    @property
    def fcd_xml_path(self) -> Path:
        return self.data_dir / f"{self.scenario}{FCD_XML_SUFFIX}"

    @property
    def sumocfg_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}{SUMOCFG_SUFFIX}"

    @property
    def traffic_generation_periods(self) -> tuple[float, ...]:
        return self._TRAFFIC_GENERATION_PERIODS[self.scenario]

    @property
    def random_seed(self) -> int:
        return self._RANDOM_SEEDS[self.scenario]

    @property
    def fcd_csv_path(self) -> Path:
        return self.data_dir / f"{self.scenario}{FCD_CSV_SUFFIX}"
