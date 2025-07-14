from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import DATA_DIR
from thesis.eta.config import SCENARIOS


@dataclass(frozen=True)
class ScenarioSpec:
    """
    A scenario specification to be used for experiments.

    Attributes:
        scenario_name (str): The name of the scenario.
        train_path (Path): The path to the training dataset.
        test_path (Path): The path to the test dataset.
    """

    scenario_name: str
    train_path: Path
    test_path: Path


def get_train_path(scenario_name: str) -> Path:
    """
    Get the path to the training dataset for a given scenario name.

    Args:
        scenario_name (str): The name of the scenario.

    Returns:
        Path: The path to the training dataset.
    """
    train_scenario_name = scenario_name.partition("-")[0]
    return DATA_DIR / f"{train_scenario_name}-train-fcd.csv"


def get_test_path(scenario_name: str) -> Path:
    """
    Get the path to the test dataset for a given scenario name.

    Args:
        scenario_name (str): The name of the scenario.

    Returns:
        Path: The path to the test dataset.
    """
    test_scenario_name = scenario_name.partition("-")[2] or scenario_name
    return DATA_DIR / f"{test_scenario_name}-test-fcd.csv"


def build_scenario_specs() -> dict[str, ScenarioSpec]:
    """
    Build the scenario specifications.

    Returns:
        dict[str, ScenarioSpec]: The scenario specifications.
    """
    specs = {}

    for scenario_name in SCENARIOS:
        specs[scenario_name] = ScenarioSpec(
            scenario_name=scenario_name,
            train_path=get_train_path(scenario_name),
            test_path=get_test_path(scenario_name),
        )

    return specs
