import logging
from dataclasses import ClassVar, dataclass
from pathlib import Path

from thesis.common.config import (
    NETWORK_BASE_FILENAME,
    NETWORK_RAIN_FILENAME,
    RANDOM_SEED_RAIN,
    RANDOM_SEED_TEST,
    RANDOM_SEED_TRAIN,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
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
        network_path (Path): Path to the network file.
        trips_path (Path): Path to the trips file.
        emission_path (Path): Path to the emission file.
        fcd_path (Path): Path to the fcd file.
        sumocfg_path (Path): Path to the sumocfg file.
        traffic_generation_periods (tuple[float, ...]): Traffic generation periods.
        random_seed (int): Random seed.
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
            f"{self.network_path=}, "
            f"{self.traffic_generation_periods=}, "
            f"{self.random_seed=})"
        )

    @property
    def network_path(self) -> Path:
        return self.simulation_dir / self._NETWORK_PATHS[self.scenario]

    @property
    def trips_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}.trips.xml"

    @property
    def emission_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}-emission.xml"

    @property
    def fcd_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}-fcd.xml"

    @property
    def sumocfg_path(self) -> Path:
        return self.simulation_dir / f"{self.scenario}.sumocfg"

    @property
    def traffic_generation_periods(self) -> tuple[float, ...]:
        return self._TRAFFIC_GENERATION_PERIODS[self.scenario]

    @property
    def random_seed(self) -> int:
        return self._RANDOM_SEEDS[self.scenario]
