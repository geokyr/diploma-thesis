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
    NETWORK_BASE,
    NETWORK_RAIN,
    SCENARIO_BASE,
    SCENARIO_RAIN,
    SCENARIOS,
    SEED_BASE,
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
        network_file (Path): The path to the network file.
        config (Path): The path to the configuration file.
        fcd_output_xml (Path): The path to the FCD output XML file.
        emission_output_xml (Path): The path to the emission output XML file.
    """

    dataset_name: str
    trips_file: Path
    traffic_generation_periods: list[float]
    seed: int
    network_file: Path
    config: Path
    fcd_output_xml: Path
    emission_output_xml: Path


def get_dataset_name(scenario_name: str, type_name: str) -> str:
    """
    Get the dataset name for a given scenario name and type name.

    Args:
        scenario_name (str): The scenario name.
        type_name (str): The type name.

    Returns:
        str: The dataset name.
    """
    return f"{scenario_name}-{type_name}"


def get_trips_file(dataset_name: str) -> Path:
    """
    Get the trips file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the trips file.
    """
    return SIMULATION_DIR / f"{dataset_name}.trips.xml"


def get_traffic_generation_periods(type_name: str) -> list[float]:
    """
    Get the traffic generation periods for a given type name.

    Args:
        type_name (str): The type name.

    Returns:
        list[float]: The traffic generation periods.
    """
    type_traffic_generation_periods = {
        TYPE_TRAIN: TRAIN_TRAFFIC_GENERATION_PERIODS,
        TYPE_TEST: TEST_TRAFFIC_GENERATION_PERIODS,
    }
    return type_traffic_generation_periods.get(type_name, [1.0])


def get_seed(scenario_name: str) -> int:
    """
    Get the seed for a given scenario name.

    Returns:
        int: The seed for the scenario name.
    """
    scenario_seeds = {
        SCENARIO_BASE: SEED_BASE,
        SCENARIO_RAIN: SEED_RAIN,
    }
    return scenario_seeds.get(scenario_name, RANDOM_SEED)


def get_network_file(scenario_name: str) -> Path:
    """
    Get the network file for a given scenario name.

    Args:
        scenario_name (str): The scenario name.

    Returns:
        Path: The path to the network file.
    """
    scenario_networks = {
        SCENARIO_BASE: NETWORK_BASE,
        SCENARIO_RAIN: NETWORK_RAIN,
    }
    return scenario_networks.get(scenario_name, NETWORK_BASE)


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

    for scenario_name in SCENARIOS:
        for type_name in TYPES:
            dataset_name = get_dataset_name(scenario_name, type_name)

            specs[dataset_name] = DatasetSpec(
                dataset_name=dataset_name,
                trips_file=get_trips_file(dataset_name),
                traffic_generation_periods=get_traffic_generation_periods(type_name),
                seed=get_seed(scenario_name),
                network_file=get_network_file(scenario_name),
                config=get_config(dataset_name),
                fcd_output_xml=get_fcd_output_xml(dataset_name),
                emission_output_xml=get_emission_output_xml(dataset_name),
            )

    return specs
