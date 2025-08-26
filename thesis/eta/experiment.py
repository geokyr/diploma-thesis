"""
ETA experiment management.
Defines experiment configurations, model persistence, and evaluation frameworks for systematic ETA prediction research.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

import joblib
from sklearn.base import BaseEstimator

from thesis.common.config import (
    DATA_DIRNAME,
    FCD_PARQUET_SUFFIX,
    LOGS_DIRNAME,
    MODELS_DIRNAME,
    OUTPUTS_DIR,
    RESULTS_DIRNAME,
    SIMULATION_DIR,
)
from thesis.eta.models import ModelType
from thesis.simulation.scenario import SimulationScenario

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
        train_path (Path): Path to the train fcd parquet file.
        test_path (Path): Path to the test fcd parquet file.
        rain_path (Path): Path to the rain fcd parquet file.
        trained_models_dir (Path): Directory for the trained models.
    """

    experiment_filename: str
    evaluation: ETAEvaluation

    _DATA_DIR: ClassVar[Path] = SIMULATION_DIR / DATA_DIRNAME
    _TRAIN_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.TRAIN}{FCD_PARQUET_SUFFIX}"
    _TEST_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.TEST}{FCD_PARQUET_SUFFIX}"
    _RAIN_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.RAIN}{FCD_PARQUET_SUFFIX}"

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
