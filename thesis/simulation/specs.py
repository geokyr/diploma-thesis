from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import (
    RANDOM_SEED,
    SIMULATION_DIR,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TYPE_TEST,
    TYPE_TRAIN,
)
from thesis.simulation.config import (
    SCENARIO_BASE,
    SCENARIO_CLOSURE,
    SCENARIO_RAIN,
    SCENARIOS,
    SEED_BASE,
    SEED_CLOSURE,
    SEED_RAIN,
    TYPES,
)


@dataclass(frozen=True)
class DatasetSpec:
    """
    A dataset specification to be used for simulation.

    Attributes:
        dataset_name (str): The name of the dataset.
        trips_file (Path): The path to the trips file.
        traffic_generation_periods (list[float]): The traffic generation periods.
        seed (int): The seed for the random number generator.
        config (Path): The path to the configuration file.
        fcd_output_xml (Path): The path to the FCD output XML file.
        emission_output_xml (Path): The path to the emission output XML file.
    """

    dataset_name: str
    trips_file: Path
    traffic_generation_periods: list[float]
    seed: int
    config: Path
    fcd_output_xml: Path
    emission_output_xml: Path


def get_dataset_name(scenario: str, type: str) -> str:
    """
    Get the dataset name for a given scenario and type of dataset.

    Args:
        scenario (str): The scenario name.
        type (str): The type of dataset.

    Returns:
        str: The dataset name.
    """
    return f"{scenario}-{type}"


def get_trips_file(dataset_name: str) -> Path:
    """
    Get the trips file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the trips file.
    """
    return SIMULATION_DIR / f"{dataset_name}.trips.xml"


def get_traffic_generation_periods(type: str) -> list[float]:
    """
    Get the traffic generation periods for a given type of dataset.

    Args:
        type (str): The type of dataset.

    Returns:
        list[float]: The traffic generation periods.
    """
    type_traffic_generation_periods = {
        TYPE_TRAIN: TRAIN_TRAFFIC_GENERATION_PERIODS,
        TYPE_TEST: TEST_TRAFFIC_GENERATION_PERIODS,
    }
    return type_traffic_generation_periods.get(type, [1.0])


def get_seed(scenario: str) -> int:
    """
    Get the seed for a given scenario.

    Returns:
        int: The seed for the scenario.
    """
    scenario_seeds = {
        SCENARIO_BASE: SEED_BASE,
        SCENARIO_CLOSURE: SEED_CLOSURE,
        SCENARIO_RAIN: SEED_RAIN,
    }
    return scenario_seeds.get(scenario, RANDOM_SEED)


def get_config(dataset_name: str) -> Path:
    """
    Get the configuration file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the configuration file.
    """
    return SIMULATION_DIR / f"{dataset_name}.sumocfg"


def get_fcd_output_xml(dataset_name: str) -> Path:
    """
    Get the FCD output XML file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the FCD output XML file.
    """
    return SIMULATION_DIR / f"{dataset_name}-fcd.xml"


def get_emission_output_xml(dataset_name: str) -> Path:
    """
    Get the emission output XML file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the emission output XML file.
    """
    return SIMULATION_DIR / f"{dataset_name}-emission.xml"


def build_dataset_specs() -> dict[str, DatasetSpec]:
    """
    Build the dataset specifications.

    Returns:
        dict[str, DatasetSpec]: The dataset specifications.
    """
    specs = {}

    for scenario in SCENARIOS:
        for type in TYPES:
            dataset_name = get_dataset_name(scenario, type)

            specs[dataset_name] = DatasetSpec(
                dataset_name=dataset_name,
                trips_file=get_trips_file(dataset_name),
                traffic_generation_periods=get_traffic_generation_periods(type),
                seed=get_seed(scenario),
                config=get_config(dataset_name),
                fcd_output_xml=get_fcd_output_xml(dataset_name),
                emission_output_xml=get_emission_output_xml(dataset_name),
            )

    return specs
