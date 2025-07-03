from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import (
    DATA_DIR,
    RANDOM_SEED,
    SIMULATION_DIR,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TYPE_TEST,
    TYPE_TRAIN,
)
from thesis.simulation.config import (
    OUTPUTS,
    SCENARIO_BASE,
    SCENARIO_CLOSURE,
    SCENARIO_RAIN,
    SCENARIOS,
    SEED_BASE,
    SEED_CLOSURE,
    SEED_RAIN,
    TYPES,
    VEHICLE_CAR,
    VEHICLE_CAR_RAIN,
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
        vehicle_type (str): The type of vehicle.
        config (Path): The path to the configuration file.
        xml_files (list[Path]): The paths to the XML files.
    """

    dataset_name: str
    trips_file: Path
    traffic_generation_periods: list[float]
    seed: int
    vehicle_type: str
    config: Path
    xml_files: list[Path]


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


def get_vehicle_type(scenario: str) -> str:
    """
    Get the vehicle type for a given scenario.

    Args:
        scenario (str): The scenario name.

    Returns:
        str: The vehicle type.
    """
    vehicle_type_scenario = {
        SCENARIO_RAIN: VEHICLE_CAR_RAIN,
    }
    return vehicle_type_scenario.get(scenario, VEHICLE_CAR)


def get_config(dataset_name: str) -> Path:
    """
    Get the configuration file for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Path: The path to the configuration file.
    """
    return SIMULATION_DIR / f"{dataset_name}.sumocfg"


def get_xml_files(dataset_name: str) -> list[Path]:
    """
    Get the XML files for a given dataset name.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        list[Path]: The paths to the XML files.
    """
    return [DATA_DIR / f"{dataset_name}-{output}.xml" for output in OUTPUTS]


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
                vehicle_type=get_vehicle_type(scenario),
                config=get_config(dataset_name),
                xml_files=get_xml_files(dataset_name),
            )

    return specs
