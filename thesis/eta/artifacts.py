import json

import joblib
from sklearn.base import BaseEstimator

from thesis.eta.config import ARTIFACTS_DIR
from thesis.logger import ETA_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME, log_file=LOG_FILES_CONFIG[ETA_LOGGER_NAME])


def save_model(model: BaseEstimator, model_name: str, scenario_name: str, experiment_name: str) -> None:
    """
    Save a model to the artifacts directory.

    Args:
        model (BaseEstimator): The machine learning model to save.
        model_name (str): The name of the model.
        scenario_name (str): The name of the scenario.
        experiment_name (str): The name of the experiment.
    """
    target_dir = ARTIFACTS_DIR / experiment_name / scenario_name
    target_dir.mkdir(parents=True, exist_ok=True)

    model_path = target_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


def save_scenario_results(results: dict[str, dict[str, float]], scenario_name: str, experiment_name: str) -> None:
    """
    Save scenario results in the artifacts directory.

    Args:
        results (dict[str, dict[str, float]]): The results dictionary to save.
        scenario_name (str): The name of the scenario.
        experiment_name (str): The name of the experiment.
    """
    target_dir = ARTIFACTS_DIR / experiment_name / scenario_name
    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = target_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Scenario results saved to {results_path}")


def save_experiment_results(results: dict[str, dict[str, dict[str, float]]], experiment_name: str) -> None:
    """
    Save experiment results in the artifacts directory.

    Args:
        results (dict[str, dict[str, dict[str, float]]]): The results dictionary to save.
        experiment_name (str): The name of the experiment.
    """
    target_dir = ARTIFACTS_DIR / experiment_name
    target_dir.mkdir(parents=True, exist_ok=True)

    results_path = target_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Experiment results saved to {results_path}")


def construct_model_results_dict(
    training_time: float, evaluation_time: float, mae: float, rmse: float, mape: float
) -> dict[str, float]:
    """
    Construct a model results dictionary.

    Args:
        training_time (float): The training time.
        evaluation_time (float): The evaluation time.
        mae (float): The MAE.
        rmse (float): The RMSE.
        mape (float): The MAPE.

    Returns:
        dict[str, float]: The model results dictionary.
    """
    return {
        "training": training_time,
        "evaluation": evaluation_time,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }
