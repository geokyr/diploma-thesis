import json
import logging
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from thesis.eta.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def initialize_experiment(experiment_name: str) -> tuple[Path, Path, Path, Path]:
    """
    Initialize the experiment directory and the necessary subdirectories.

    Args:
        experiment_name (str): The name of the experiment.

    Returns:
        tuple[Path, Path, Path, Path]: The artifacts, logs, plots, and results directories.
    """
    logger.info(f"Initializing experiment {experiment_name}")

    experiment_dir = OUTPUTS_DIR / experiment_name
    artifacts_dir = experiment_dir / "artifacts"
    logs_dir = experiment_dir / "logs"
    plots_dir = experiment_dir / "plots"
    results_dir = experiment_dir / "results"

    for dir in [artifacts_dir, logs_dir, plots_dir, results_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    return artifacts_dir, logs_dir, plots_dir, results_dir


def save_model(model: BaseEstimator, model_name: str, scenario_name: str, artifacts_dir: Path) -> None:
    """
    Save model in the artifacts directory.

    Args:
        model (BaseEstimator): The machine learning model to save.
        model_name (str): The name of the model.
        scenario_name (str): The name of the scenario.
        artifacts_dir (Path): The directory to save the model to.
    """
    target_dir = artifacts_dir / scenario_name
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


def save_results(results: dict[str, dict[str, float]], scenario_name: str, results_dir: Path) -> None:
    """
    Save results in the results directory.

    Args:
        results (dict[str, dict[str, float]]): The results dictionary to save.
        scenario_name (str): The name of the scenario.
        results_dir (Path): The directory to save the results to.
    """
    results_path = results_dir / f"{scenario_name}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
