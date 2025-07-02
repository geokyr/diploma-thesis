from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import (
    DATA_DIR,
    SIMULATION_DIR,
    TEST_SEED,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_SEED,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TYPE_TRAIN,
)
from thesis.simulation.config import (
    FIXED_ROUTES_FILE,
    OUTPUTS,
    SCENARIO_RAIN,
    SCENARIOS,
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
        fixed_routes_file (Path | None): The path to the fixed routes file.
        config (Path): The path to the configuration file.
        xml_files (list[Path]): The paths to the XML files.
    """

    dataset_name: str
    trips_file: Path
    traffic_generation_periods: list[float]
    seed: int
    vehicle_type: str
    fixed_routes_file: Path | None
    config: Path
    xml_files: list[Path]


def build_dataset_specs() -> dict[str, DatasetSpec]:
    """
    Build the dataset specifications.

    Returns:
        dict[str, DatasetSpec]: The dataset specifications.
    """
    specs = {}

    for scenario in SCENARIOS:
        for type in TYPES:
            dataset_name = f"{scenario}-{type}"

            specs[dataset_name] = DatasetSpec(
                dataset_name=dataset_name,
                trips_file=SIMULATION_DIR / f"{dataset_name}.trips.xml",
                traffic_generation_periods=(
                    TRAIN_TRAFFIC_GENERATION_PERIODS if type == TYPE_TRAIN else TEST_TRAFFIC_GENERATION_PERIODS
                ),
                seed=TRAIN_SEED if type == TYPE_TRAIN else TEST_SEED,
                vehicle_type=VEHICLE_CAR_RAIN if scenario == SCENARIO_RAIN else VEHICLE_CAR,
                fixed_routes_file=None if type == TYPE_TRAIN else FIXED_ROUTES_FILE,
                config=SIMULATION_DIR / f"{dataset_name}.sumocfg",
                xml_files=[DATA_DIR / f"{dataset_name}-{output}.xml" for output in OUTPUTS],
            )

    return specs
