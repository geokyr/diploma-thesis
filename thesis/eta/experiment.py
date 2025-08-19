import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from thesis.common.config import (
    DATA_DIRNAME,
    LOGS_DIRNAME,
    MODELS_DIRNAME,
    OUTPUTS_DIR,
    RESULTS_DIRNAME,
    RESULTS_FILENAME,
    SIMULATION_DIR,
)
from thesis.eta.models import ModelType

logger = logging.getLogger(__name__)


class ETAEvaluation(StrEnum):
    """
    ETA Evaluations.

    Attributes:
        RESEARCH: Evaluate via cross-validation on training data.
        STABLE: Evaluate on a clean, held-out test set.
        DRIFT: Evaluate on drift-affected data to quantify degradation.
        RETRAIN: Evaluate after retraining on drift-affected data to assess recovery.
    """

    RESEARCH = "research"
    STABLE = "stable"
    DRIFT = "drift"
    RETRAIN = "retrain"


@dataclass(frozen=True, slots=True)
class ETAExperiment:
    """
    An ETA experiment.

    Attributes:
        experiment_filename (str): Name of the file that contains the ETA experiment.
        evaluation (ETAEvaluation): Evaluation of the experiment.

    Properties:
        name (str): Name of the ETA experiment.
        experiment_dir (Path): Directory for the ETA experiment.
        models_dir (Path): Subdirectory for the models.
        logs_dir (Path): Subdirectory for the logs.
        results_dir (Path): Subdirectory for the results.
        train_path (Path): Path to the train fcd csv file.
        test_path (Path): Path to the test fcd csv file.
        rain_path (Path): Path to the rain fcd csv file.
        trained_models_dir (Path): Directory for the trained models.
    """

    experiment_filename: str
    evaluation: ETAEvaluation

    _DATA_DIR: ClassVar[Path] = SIMULATION_DIR / DATA_DIRNAME
    _TRAIN_PATH: ClassVar[Path] = _DATA_DIR / "train-fcd.csv"
    _TEST_PATH: ClassVar[Path] = _DATA_DIR / "test-fcd.csv"
    _RAIN_PATH: ClassVar[Path] = _DATA_DIR / "rain-fcd.csv"

    _TRAINED_MODELS_DIR: ClassVar[Path] = OUTPUTS_DIR / ETAEvaluation.STABLE / MODELS_DIRNAME

    def __post_init__(self) -> None:
        for dir in [self.models_dir, self.logs_dir, self.results_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.experiment_filename=}, "
            f"{self.evaluation=}, "
            f"{self.name=}, "
            f"{self.experiment_dir=}, "
            f"{self.models_dir=}, "
            f"{self.logs_dir=}, "
            f"{self.results_dir=}, "
            f"{self.train_path=}, "
            f"{self.test_path=}, "
            f"{self.rain_path=}, "
            f"{self.trained_models_dir=})"
        )

    @property
    def name(self) -> str:
        return Path(self.experiment_filename).stem

    @property
    def experiment_dir(self) -> Path:
        return OUTPUTS_DIR / self.name

    @property
    def models_dir(self) -> Path:
        return self.experiment_dir / MODELS_DIRNAME

    @property
    def logs_dir(self) -> Path:
        return self.experiment_dir / LOGS_DIRNAME

    @property
    def results_dir(self) -> Path:
        return self.experiment_dir / RESULTS_DIRNAME

    @property
    def train_path(self) -> Path:
        return self._TRAIN_PATH

    @property
    def test_path(self) -> Path:
        return self._TEST_PATH

    @property
    def rain_path(self) -> Path:
        return self._RAIN_PATH

    @property
    def trained_models_dir(self) -> Path:
        return self._TRAINED_MODELS_DIR


def build_model_results(
    training_results: dict[str, float], prediction_results: dict[str, float], evaluation_results: dict[str, float]
) -> dict[str, float]:
    """
    Build the model results by merging the training, prediction and evaluation results.

    Args:
        training_results (dict[str, float]): Training results.
        prediction_results (dict[str, float]): Prediction results.
        evaluation_results (dict[str, float]): Evaluation results.

    Returns:
        dict[str, float]: Merged results.
    """
    return {
        **training_results,
        **prediction_results,
        **evaluation_results,
    }


def build_cv_results(per_fold_results: list[dict[str, float]]) -> dict[str, float]:
    """
    Calculate mean and standard deviation of cross-validation results.

    Args:
        per_fold_results (list[dict[str, float]]): List of dictionaries containing results for each fold.

    Returns:
        dict[str, float]: Dictionary containing mean, std and per-fold results.
    """
    df_results = pd.DataFrame(per_fold_results)
    cv_mean = df_results.mean().to_dict()
    cv_std = df_results.std().to_dict()

    return {
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "per_fold": per_fold_results,
    }


def save_model(model: BaseEstimator, model_type: ModelType, models_dir: Path) -> None:
    """
    Save model in the models directory.

    Args:
        model (BaseEstimator): Machine learning model to save.
        model_type (ModelType): Type of the model.
        models_dir (Path): Directory to save the model to.
    """
    model_path = models_dir / f"{model_type}.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


def load_model(model_type: ModelType, models_dir: Path) -> BaseEstimator:
    """
    Load model from the models directory.

    Args:
        model_type (ModelType): Type of the model.
        models_dir (Path): Directory to load the model from.

    Returns:
        BaseEstimator: Loaded machine learning model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = models_dir / f"{model_type}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    return model


def save_results(results: dict[str, dict[str, float]], results_dir: Path) -> None:
    """
    Save results in the results directory.

    Args:
        results (dict[str, dict[str, float]]): Results dictionary to save.
        results_dir (Path): Directory to save the results to.
    """
    results_path = results_dir / RESULTS_FILENAME
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_path}")
