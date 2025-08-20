"""
ETA results management.
Defines data structures for storing, analyzing and visualizing ETA model results.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thesis.common.config import OUTPUTS_DIR, RESULTS_DIRNAME, RESULTS_FILENAME
from thesis.eta.models import ModelType

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingResults:
    """
    Training results for a model.

    Attributes:
        training_time (float): Time taken to train the model in seconds.
    """

    training_time: float


@dataclass(frozen=True, slots=True)
class PredictionResults:
    """
    Prediction results for a model.

    Attributes:
        prediction_time (float): Time taken to make predictions in seconds.
    """

    prediction_time: float


@dataclass(frozen=True, slots=True)
class EvaluationResults:
    """
    Evaluation results for a model.

    Attributes:
        evaluation_time (float): Time taken to evaluate predictions in seconds.
        mae (float): Mean Absolute Error.
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
        mape (float): Mean Absolute Percentage Error.
        r2 (float): R-squared score.
    """

    evaluation_time: float
    mae: float
    mse: float
    rmse: float
    mape: float
    r2: float


@dataclass(frozen=True, slots=True)
class ModelResults:
    """
    Complete results for a model including training, prediction, and evaluation results.

    Attributes:
        training_results (TrainingResults): Training results.
        prediction_results (PredictionResults): Prediction results.
        evaluation_results (EvaluationResults): Evaluation results.
    """

    training_results: TrainingResults
    prediction_results: PredictionResults
    evaluation_results: EvaluationResults

    def to_flat_dict(self) -> dict[str, float]:
        """
        Convert to a flat dictionary for JSON serialization.

        Returns:
            dict[str, float]: Flattened dictionary with all metrics.
        """
        return {
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "evaluation_time": self.evaluation_time,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2,
        }

    @property
    def training_time(self) -> float:
        return self.training_results.training_time

    @property
    def prediction_time(self) -> float:
        return self.prediction_results.prediction_time

    @property
    def evaluation_time(self) -> float:
        return self.evaluation_results.evaluation_time

    @property
    def mae(self) -> float:
        return self.evaluation_results.mae

    @property
    def mse(self) -> float:
        return self.evaluation_results.mse

    @property
    def rmse(self) -> float:
        return self.evaluation_results.rmse

    @property
    def mape(self) -> float:
        return self.evaluation_results.mape

    @property
    def r2(self) -> float:
        return self.evaluation_results.r2

    @classmethod
    def from_flat_dict(cls, data: dict[str, float]) -> "ModelResults":
        """
        Create ModelResults from a flat dictionary.

        Args:
            data (dict[str, float]): Dictionary with flattened metrics.

        Returns:
            ModelResults: New ModelResults instance.
        """
        return cls(
            training_results=TrainingResults(training_time=data["training_time"]),
            prediction_results=PredictionResults(prediction_time=data["prediction_time"]),
            evaluation_results=EvaluationResults(
                evaluation_time=data["evaluation_time"],
                mae=data["mae"],
                mse=data["mse"],
                rmse=data["rmse"],
                mape=data["mape"],
                r2=data["r2"],
            ),
        )

    @classmethod
    def from_components(
        cls,
        training_results: TrainingResults,
        prediction_results: PredictionResults,
        evaluation_results: EvaluationResults,
    ) -> "ModelResults":
        """
        Create ModelResults from individual result components.

        Args:
            training_results (TrainingResults): Training results.
            prediction_results (PredictionResults): Prediction results.
            evaluation_results (EvaluationResults): Evaluation results.

        Returns:
            ModelResults: New ModelResults instance.
        """
        return cls(
            training_results=training_results,
            prediction_results=prediction_results,
            evaluation_results=evaluation_results,
        )


@dataclass(frozen=True, slots=True)
class CVResults:
    """
    Cross-validation results containing mean results and per-fold results.

    Attributes:
        cv_mean (ModelResults): Mean results across all folds.
        per_fold (list[ModelResults]): Results for each individual fold.
    """

    cv_mean: ModelResults
    per_fold: list[ModelResults]

    def to_flat_dict(self) -> dict[str, float | list[dict[str, float]]]:
        """
        Convert to a flat dictionary for JSON serialization.

        Returns:
            dict[str, float | list[dict[str, float]]]: Flattened dictionary with mean metrics and per_fold data.
        """
        return {
            "training_time": self.cv_mean.training_time,
            "prediction_time": self.cv_mean.prediction_time,
            "evaluation_time": self.cv_mean.evaluation_time,
            "mae": self.cv_mean.mae,
            "mse": self.cv_mean.mse,
            "rmse": self.cv_mean.rmse,
            "mape": self.cv_mean.mape,
            "r2": self.cv_mean.r2,
            "per_fold": [fold_result.to_flat_dict() for fold_result in self.per_fold],
        }

    @property
    def training_time(self) -> float:
        return self.cv_mean.training_time

    @property
    def prediction_time(self) -> float:
        return self.cv_mean.prediction_time

    @property
    def evaluation_time(self) -> float:
        return self.cv_mean.evaluation_time

    @property
    def mae(self) -> float:
        return self.cv_mean.mae

    @property
    def mse(self) -> float:
        return self.cv_mean.mse

    @property
    def rmse(self) -> float:
        return self.cv_mean.rmse

    @property
    def mape(self) -> float:
        return self.cv_mean.mape

    @property
    def r2(self) -> float:
        return self.cv_mean.r2

    @classmethod
    def from_per_fold_results(cls, per_fold_results: list[ModelResults]) -> "CVResults":
        """
        Create CVResults from per-fold ModelResults.

        Args:
            per_fold_results (list[ModelResults]): List of ModelResults containing results for each fold.

        Returns:
            CVResults: Cross-validation results containing mean results and per-fold results.
        """
        flattened_data = [result.to_flat_dict() for result in per_fold_results]
        df_results = pd.DataFrame(flattened_data)
        cv_mean_dict = df_results.mean().to_dict()

        cv_mean = ModelResults.from_flat_dict(cv_mean_dict)

        return cls(
            cv_mean=cv_mean,
            per_fold=per_fold_results,
        )


def save_results(results: dict[ModelType, ModelResults | CVResults], results_dir: Path) -> None:
    """
    Save results in the results directory.

    Args:
        results (dict[ModelType, ModelResults | CVResults]): Results dictionary to save.
        results_dir (Path): Directory to save the results to.
    """
    results_path = results_dir / RESULTS_FILENAME

    results_dict = {}
    for model_type, result in results.items():
        results_dict[model_type] = result.to_flat_dict()

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    logger.info(f"Results saved to {results_path}")


def load_results(experiment_path: Path) -> dict[ModelType, ModelResults | CVResults]:
    """
    Load results from an experiment directory.

    Args:
        experiment_path (Path): Path to the experiment directory.

    Returns:
        dict[ModelType, ModelResults | CVResults]: Dictionary containing the experiment results.
    """
    results_dir = experiment_path / RESULTS_DIRNAME
    if not results_dir.is_dir():
        return {}

    json_file = results_dir / RESULTS_FILENAME
    if not json_file.exists():
        return {}

    with open(json_file, "r") as f:
        return json.load(f)


def build_model_results(
    training_results: TrainingResults, prediction_results: PredictionResults, evaluation_results: EvaluationResults
) -> ModelResults:
    """
    Build the model results by merging the training, prediction and evaluation results.

    Args:
        training_results (TrainingResults): Training results.
        prediction_results (PredictionResults): Prediction results.
        evaluation_results (EvaluationResults): Evaluation results.

    Returns:
        ModelResults: Merged results.
    """
    return ModelResults.from_components(training_results, prediction_results, evaluation_results)


def build_cv_results(per_fold_results: list[ModelResults]) -> CVResults:
    """
    Calculate mean of cross-validation results.

    Args:
        per_fold_results (list[ModelResults]): List of ModelResults containing results for each fold.

    Returns:
        CVResults: Cross-validation results containing mean results and per-fold results.
    """
    return CVResults.from_per_fold_results(per_fold_results)


def load_all_results() -> pd.DataFrame:
    """
    Load all experiment results and return as a clean DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing all experiment results.
    """
    records = []

    experiment_dirs = [dir for dir in OUTPUTS_DIR.iterdir() if dir.is_dir()]
    for experiment_dir in experiment_dirs:
        experiment_name = experiment_dir.name
        experiment_results = load_results(experiment_dir)

        if experiment_results:
            for model, metrics in experiment_results.items():
                row = {
                    "experiment": experiment_name,
                    "model": model,
                    **metrics,
                }
                records.append(row)

    df = pd.DataFrame(records)

    if "per_fold" in df.columns:
        df.drop(columns=["per_fold"], inplace=True)

    df.sort_values(["experiment", "model"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Loaded {len(df)} experiment results as DataFrame")

    return df
