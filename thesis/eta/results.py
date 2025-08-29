"""
ETA results management.
Defines data structures for storing, analyzing and visualizing ETA model results.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns

from thesis.common.config import (
    OUTPUTS_DIR,
    RESEARCH_RESULTS_FILENAME,
    RESULTS_DIRNAME,
    RESULTS_FILENAME,
    TUNING_RESULTS_FILENAME,
)
from thesis.eta.experiment import ETAEvaluation
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
        mape (float): Mean Absolute Percentage Error.
    """

    evaluation_time: float
    mae: float
    mape: float


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
            "mape": self.mape,
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
    def mape(self) -> float:
        return self.evaluation_results.mape

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
                mape=data["mape"],
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
            "mape": self.cv_mean.mape,
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
    def mape(self) -> float:
        return self.cv_mean.mape

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


def load_research_results() -> pd.DataFrame:
    """
    Load research experiment results and return as a clean DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing research experiment results.
    """
    df = load_all_results()

    evaluations = [evaluation.value for evaluation in ETAEvaluation]
    evaluations.remove(ETAEvaluation.RESEARCH.value)

    filtered_df = df[~df["experiment"].isin(evaluations)]

    logger.info(f"Filtered to {len(filtered_df)} research experiment results (excluded evaluations: {evaluations})")

    return filtered_df


def load_research_results_without_baselines() -> pd.DataFrame:
    """
    Load research experiment results excluding baseline models and return as a clean DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing research experiment results without baseline models.
    """
    df = load_research_results()

    filtered_df = df[df["model"] != ModelType.LINEAR_REGRESSION]

    logger.info(
        f"Filtered to {len(filtered_df)} research experiment results without baselines (excluded model: {ModelType.LINEAR_REGRESSION})"
    )

    return filtered_df


def save_research_results(results_df: pd.DataFrame, results_dir: Path) -> None:
    """
    Save research experiment results to a CSV file.

    Args:
        results_df (pd.DataFrame): DataFrame with results.
        results_dir (Path): Directory to save the results.
    """
    results_path = results_dir / RESEARCH_RESULTS_FILENAME
    results_df.to_csv(results_path, index=False)
    logger.info(f"Research experiment results saved to {results_path}")


def plot_metric_by_experiment(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, decimal_places: int, plots_dir: Path
) -> None:
    """
    Plot average metric values by experiment.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        plots_dir (Path): Directory to save the plots.
    """
    plot_path = plots_dir / f"{metric_column}_by_experiment.png"

    metric_by_exp = df.groupby("experiment")[metric_column].mean()

    num_experiments = len(metric_by_exp)
    width = 8
    height = max(6, num_experiments * 0.4)

    plt.figure(figsize=(width, height))
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_by_exp)))
    plt.barh(range(len(metric_by_exp)), metric_by_exp.values, color=colors)
    plt.title(f"Average {metric_name} by Experiment", fontweight="bold")
    plt.xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    plt.ylabel("Experiment")
    plt.yticks(range(len(metric_by_exp)), metric_by_exp.index)
    plt.gca().invert_yaxis()

    min_val, max_val = metric_by_exp.values.min(), metric_by_exp.values.max()
    range_val = max_val - min_val
    plt.xlim(min_val - range_val * 0.2, max_val + range_val * 0.2)

    for i, v in enumerate(metric_by_exp.values):
        label = f"{v:.{decimal_places}f}{metric_unit}"
        plt.text(v + range_val * 0.02, i, label, ha="left", va="center", fontweight="bold", color="black")

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"{metric_name} by experiment plot saved to {plot_path}")


def plot_metric_by_model(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, decimal_places: int, plots_dir: Path
) -> None:
    """
    Plot average metric values by model.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        plots_dir (Path): Directory to save the plots.
    """
    plot_path = plots_dir / f"{metric_column}_by_model.png"

    metric_by_model = df.groupby("model")[metric_column].mean()

    num_models = len(metric_by_model)
    width = 8
    height = max(4, num_models * 0.4)

    plt.figure(figsize=(width, height))
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_by_model)))
    plt.barh(range(len(metric_by_model)), metric_by_model.values, color=colors)
    plt.title(f"Average {metric_name} by Model", fontweight="bold")
    plt.xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    plt.ylabel("Model")
    plt.yticks(range(len(metric_by_model)), metric_by_model.index)
    plt.gca().invert_yaxis()

    min_val, max_val = metric_by_model.values.min(), metric_by_model.values.max()
    range_val = max_val - min_val
    plt.xlim(min_val - range_val * 0.2, max_val + range_val * 0.2)

    for i, v in enumerate(metric_by_model.values):
        label = f"{v:.{decimal_places}f}{metric_unit}"
        plt.text(v + range_val * 0.02, i, label, ha="left", va="center", fontweight="bold", color="black")

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"{metric_name} by model plot saved to {plot_path}")


def plot_metric_heatmap(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, decimal_places: int, plots_dir: Path
) -> None:
    """
    Plot metric heatmap of experiment vs model.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        plots_dir (Path): Directory to save the plots.
    """
    plot_path = plots_dir / f"{metric_column}_heatmap.png"

    metric_pivot = df.pivot(index="experiment", columns="model", values=metric_column)

    num_experiments = len(metric_pivot.index)
    num_models = len(metric_pivot.columns)
    width = max(8, num_models * 1.2)
    height = max(4, num_experiments * 0.5)

    plt.figure(figsize=(width, height))
    cbar_label = f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}"
    sns.heatmap(
        metric_pivot,
        annot=True,
        fmt=f".{decimal_places}f",
        cmap="viridis",
        cbar_kws={"label": cbar_label},
    )
    plt.title(f"{metric_name} Heatmap: Experiment vs Model", fontweight="bold")
    plt.xlabel("Model")
    plt.ylabel("Experiment")

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"{metric_name} heatmap saved to {plot_path}")


def plot_metric_distribution(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, plots_dir: Path
) -> None:
    """
    Plot metric distribution by experiment using boxplots.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        plots_dir (Path): Directory to save the plots.
    """
    plot_path = plots_dir / f"{metric_column}_distribution.png"

    experiments = sorted(df["experiment"].unique())

    num_experiments = len(experiments)
    width = 8
    height = max(6, num_experiments * 0.4)

    plt.figure(figsize=(width, height))
    sns.boxplot(
        data=df,
        y="experiment",
        x=metric_column,
        hue="experiment",
        order=experiments,
        palette=[plt.cm.viridis(i / len(experiments)) for i in range(len(experiments))],
        legend=False,
        orient="h",
    )
    plt.title(f"{metric_name} Distribution by Experiment", fontweight="bold")
    plt.xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    plt.ylabel("Experiment")

    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logger.info(f"{metric_name} distribution plot saved to {plot_path}")


def run_metric_analysis(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, decimal_places: int, plots_dir: Path
) -> None:
    """
    Run a comprehensive analysis for a metric by generating multiple plots.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        plots_dir (Path): Directory to save the plots.
    """
    logger.info(f"Running {metric_name} analysis")

    if metric_unit == "%":
        df[metric_column] = df[metric_column] * 100

    plot_metric_by_experiment(df, metric_column, metric_name, metric_unit, decimal_places, plots_dir)
    plot_metric_by_model(df, metric_column, metric_name, metric_unit, decimal_places, plots_dir)
    plot_metric_heatmap(df, metric_column, metric_name, metric_unit, decimal_places, plots_dir)
    plot_metric_distribution(df, metric_column, metric_name, metric_unit, plots_dir)

    logger.info(f"Completed {metric_name} analysis")


def build_tuning_results(
    study: optuna.Study,
) -> dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]]:
    """
    Build tuning results dictionary from Optuna study.

    Args:
        study (optuna.Study): Completed Optuna study.

    Returns:
        dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]]: Dictionary containing study results and trial data.
    """
    results = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "model": study.user_attrs["model"],
    }

    trials_data = []
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "user_attrs": trial.user_attrs,
        }
        trials_data.append(trial_data)

    results["trials"] = trials_data

    return results


def save_tuning_results(
    results: dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]],
    results_dir: Path,
) -> None:
    """
    Save tuning results to JSON file.

    Args:
        results (dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]]): Tuning results dictionary.
        results_dir (Path): Directory to save results to.
    """
    results_path = results_dir / TUNING_RESULTS_FILENAME
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to {results_path}")


def load_tuning_results(
    experiment_path: Path,
) -> dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]]:
    """
    Load tuning results from an experiment directory.

    Args:
        experiment_path (Path): Path to the experiment directory.

    Returns:
        dict[str, float | str | dict[str, float] | list[dict[str, float | str | dict[str, float]]]]: Dictionary containing the tuning experiment results.
    """
    results_dir = experiment_path / RESULTS_DIRNAME
    if not results_dir.is_dir():
        return {}

    tuning_file = results_dir / TUNING_RESULTS_FILENAME
    if not tuning_file.exists():
        return {}

    with open(tuning_file, "r") as f:
        return json.load(f)


def load_all_tuning_results() -> dict[str, pd.DataFrame]:
    """
    Load all tuning experiment results and return as DataFrames grouped by model.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with model names as keys and DataFrames as values.
    """
    results = {}

    tuning_experiment_dirs = [dir for dir in OUTPUTS_DIR.iterdir() if dir.is_dir() and "tuning" in dir.name]
    for tuning_experiment_dir in tuning_experiment_dirs:
        experiment_name = tuning_experiment_dir.name
        tuning_experiment_results = load_tuning_results(tuning_experiment_dir)

        if not tuning_experiment_results:
            continue

        records = []
        model = tuning_experiment_results["model"]

        for trial in tuning_experiment_results["trials"]:
            if trial["state"] != "COMPLETE":
                continue

            user_attrs = trial["user_attrs"]

            records.append(
                {
                    "experiment": experiment_name,
                    "model": model,
                    "trial_number": trial["number"],
                    "mae": trial["value"],
                    "mape": user_attrs["mape"],
                    "training_time": user_attrs["training_time"],
                    "prediction_time": user_attrs["prediction_time"],
                    "evaluation_time": user_attrs["evaluation_time"],
                    "total_time": (
                        user_attrs["training_time"] + user_attrs["prediction_time"] + user_attrs["evaluation_time"]
                    ),
                    **trial["params"],
                }
            )

        if not records:
            continue

        df = pd.DataFrame(records)

        if model not in results:
            results[model] = df
        else:
            results[model] = pd.concat([results[model], df])

    constant_parameters = {
        ModelType.CATBOOST_REGRESSOR: {"border_count": 255},
        ModelType.LIGHTGBM_REGRESSOR: {"reg_alpha": 0.0},
    }

    final_results = {}
    for model, df in results.items():
        if model in constant_parameters:
            for parameter, value in constant_parameters[model].items():
                if parameter in df.columns:
                    df[parameter] = df[parameter].fillna(value)

        final_results[model] = df.reset_index(drop=True)

    return final_results
