"""Managing, storing, analyzing and visualizing ETA model results."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from thesis.common.config import (
    OUTPUTS_DIR,
    RESEARCH_RESULTS_FILENAME,
    RESULTS_DIRNAME,
    RESULTS_FILENAME,
    TUNING_RESULTS_FILENAME,
)
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
        """Time taken to train the model in seconds."""
        return self.training_results.training_time

    @property
    def prediction_time(self) -> float:
        """Time taken to make predictions in seconds."""
        return self.prediction_results.prediction_time

    @property
    def evaluation_time(self) -> float:
        """Time taken to evaluate predictions in seconds."""
        return self.evaluation_results.evaluation_time

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return self.evaluation_results.mae

    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
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
        """Time taken to train the model in seconds."""
        return self.cv_mean.training_time

    @property
    def prediction_time(self) -> float:
        """Time taken to make predictions in seconds."""
        return self.cv_mean.prediction_time

    @property
    def evaluation_time(self) -> float:
        """Time taken to evaluate predictions in seconds."""
        return self.cv_mean.evaluation_time

    @property
    def mae(self) -> float:
        """Mean Absolute Error."""
        return self.cv_mean.mae

    @property
    def mape(self) -> float:
        """Mean Absolute Percentage Error."""
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

    keywords = ["transform", "features", "research"]
    pattern = "|".join(keywords)

    filtered_df = df[df["experiment"].str.contains(pattern, case=False, regex=True)]

    logger.info(f"Filtered to {len(filtered_df)} research experiment results")

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
    df: pd.DataFrame,
    metric_column: str,
    metric_name: str,
    metric_unit: str,
    decimal_places: int,
    ax: plt.Axes,
) -> None:
    """
    Plot average metric values by experiment on a given axis.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        ax (plt.Axes): Matplotlib axis to plot on.
    """
    metric_by_exp = df.groupby("experiment")[metric_column].mean()
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_by_exp)))

    ax.barh(range(len(metric_by_exp)), metric_by_exp.to_numpy(), color=colors)
    ax.set_title(f"Average {metric_name} by Experiment", fontweight="bold")
    ax.set_xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    ax.set_ylabel("Experiment")
    ax.set_yticks(range(len(metric_by_exp)))
    ax.set_yticklabels(metric_by_exp.index)
    ax.invert_yaxis()

    min_val, max_val = metric_by_exp.to_numpy().min(), metric_by_exp.to_numpy().max()
    range_val = max_val - min_val
    ax.set_xlim(min_val - range_val * 0.2, max_val + range_val * 0.2)

    for i, v in enumerate(metric_by_exp.to_numpy()):
        label = f"{v:.{decimal_places}f}{metric_unit}"
        ax.text(v + range_val * 0.02, i, label, ha="left", va="center", fontweight="bold", color="black")


def plot_metric_by_model(
    df: pd.DataFrame,
    metric_column: str,
    metric_name: str,
    metric_unit: str,
    decimal_places: int,
    ax: plt.Axes,
) -> None:
    """
    Plot average metric values by model on a given axis.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        ax (plt.Axes): Matplotlib axis to plot on.
    """
    metric_by_model = df.groupby("model")[metric_column].mean()
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_by_model)))

    ax.barh(range(len(metric_by_model)), metric_by_model.to_numpy(), color=colors)
    ax.set_title(f"Average {metric_name} by Model", fontweight="bold")
    ax.set_xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    ax.set_ylabel("Model")
    ax.set_yticks(range(len(metric_by_model)))
    ax.set_yticklabels(metric_by_model.index)
    ax.invert_yaxis()

    min_val, max_val = metric_by_model.to_numpy().min(), metric_by_model.to_numpy().max()
    range_val = max_val - min_val
    ax.set_xlim(min_val - range_val * 0.2, max_val + range_val * 0.2)

    for i, v in enumerate(metric_by_model.to_numpy()):
        label = f"{v:.{decimal_places}f}{metric_unit}"
        ax.text(v + range_val * 0.02, i, label, ha="left", va="center", fontweight="bold", color="black")


def plot_metric_heatmap(
    df: pd.DataFrame,
    metric_column: str,
    metric_name: str,
    metric_unit: str,
    decimal_places: int,
    ax: plt.Axes,
) -> None:
    """
    Plot metric heatmap of experiment vs model on a given axis.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        decimal_places (int): Number of decimal places to show for the metric.
        ax (plt.Axes): Matplotlib axis to plot on.
    """
    metric_pivot = df.pivot(index="experiment", columns="model", values=metric_column)
    cbar_label = f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}"

    sns.heatmap(
        metric_pivot,
        annot=True,
        fmt=f".{decimal_places}f",
        cmap="viridis",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_title(f"{metric_name} Heatmap: Experiment vs Model", fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Experiment")


def plot_metric_distribution(
    df: pd.DataFrame,
    metric_column: str,
    metric_name: str,
    metric_unit: str,
    ax: plt.Axes,
) -> None:
    """
    Plot metric distribution by experiment using boxplots on a given axis.

    Args:
        df (pd.DataFrame): DataFrame with results.
        metric_column (str): Column name for the metric to analyze.
        metric_name (str): Display name for the metric.
        metric_unit (str): Unit of the metric.
        ax (plt.Axes): Matplotlib axis to plot on.
    """
    experiments = sorted(df["experiment"].unique())

    sns.boxplot(
        data=df,
        y="experiment",
        x=metric_column,
        hue="experiment",
        order=experiments,
        palette=[plt.cm.viridis(i / len(experiments)) for i in range(len(experiments))],
        legend=False,
        orient="h",
        ax=ax,
    )

    ax.set_title(f"{metric_name} Distribution by Experiment", fontweight="bold")
    ax.set_xlabel(f"{metric_name}{' (' + metric_unit + ')' if metric_unit else ''}")
    ax.set_ylabel("Experiment")


def run_metric_analysis(
    df: pd.DataFrame, metric_column: str, metric_name: str, metric_unit: str, decimal_places: int, plots_dir: Path
) -> None:
    """
    Run a comprehensive analysis for a metric by generating a 2x2 subplot with all analysis types.

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

    plot_path = plots_dir / f"{metric_column}_analysis.png"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    plot_metric_by_experiment(df, metric_column, metric_name, metric_unit, decimal_places, ax1)
    plot_metric_by_model(df, metric_column, metric_name, metric_unit, decimal_places, ax2)
    plot_metric_heatmap(df, metric_column, metric_name, metric_unit, decimal_places, ax3)
    plot_metric_distribution(df, metric_column, metric_name, metric_unit, ax4)

    fig.suptitle(f"{metric_name} Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

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


def load_all_tuning_results() -> dict[ModelType, pd.DataFrame]:
    """
    Load all tuning experiment results and return as DataFrames grouped by model.

    Returns:
        dict[ModelType, pd.DataFrame]: Dictionary with model names as keys and DataFrames as values.
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

    final_results = {}
    for model, df in results.items():
        final_results[model] = df.reset_index(drop=True)

    logger.info(f"Loaded tuning results for {len(final_results)} models")

    return final_results


def save_all_tuning_results(results: dict[ModelType, pd.DataFrame], results_dir: Path) -> None:
    """
    Save all tuning results to a CSV file.

    Args:
        results (dict[ModelType, pd.DataFrame]): Dictionary with model names as keys and DataFrames as values.
        results_dir (Path): Directory to save results to.
    """
    results_path = results_dir / "tuning_results.csv"

    all_data = []
    for _, df in results.items():
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(results_path, index=False)

    logger.info(f"Tuning results saved to {results_path}")


def plot_boxplot_by_model(df: pd.DataFrame, column: str, title: str, ylabel: str, ax: plt.Axes) -> None:
    """
    Plot a boxplot for a given column grouped by model on a given axis.

    Args:
        df (pd.DataFrame): DataFrame with model results
        column (str): Column name to plot
        title (str): Plot title
        ylabel (str): Y-axis label
        ax (plt.Axes): Matplotlib axis to plot on
    """
    df_plot = df.copy()
    if "%" in ylabel:
        df_plot[column] = df_plot[column] * 100

    df_plot.boxplot(column=column, by="model", ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)


def plot_mae_convergence(results: dict[str, pd.DataFrame], ax: plt.Axes) -> None:
    """
    Plot MAE convergence over trial numbers for all models on a given axis.

    Args:
        results (dict[str, pd.DataFrame]): Dictionary with model names as keys and DataFrames as values.
        ax (plt.Axes): Matplotlib axis to plot on.
    """
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    all_final_values = []
    for i, (model_name, df) in enumerate(results.items()):
        df_sorted = df.sort_values(["experiment", "trial_number"])

        continuous_trial_numbers = []
        trial_counter = 0
        for experiment in df_sorted["experiment"].unique():
            experiment_trials = df_sorted[df_sorted["experiment"] == experiment]
            experiment_trial_numbers = list(range(trial_counter, trial_counter + len(experiment_trials)))
            continuous_trial_numbers.extend(experiment_trial_numbers)
            trial_counter += len(experiment_trials)

        mae_values = df_sorted["mae"].to_numpy()
        best_mae_so_far = np.minimum.accumulate(mae_values)
        all_final_values.extend(best_mae_so_far)
        ax.plot(continuous_trial_numbers, best_mae_so_far, label=model_name, color=colors[i], linewidth=2.5)

    y_min = min(all_final_values)
    y_max = max(all_final_values)
    y_range = y_max - y_min

    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)
    ax.set_title("Best MAE Over Trials", fontweight="bold")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Best MAE So Far")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)


def run_tuning_analysis(results: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    """
    Create overview plots of model performance and timing.

    Args:
        results (dict[str, pd.DataFrame]): Dictionary with model names as keys and DataFrames as values.
        plots_dir (Path): Directory to save the plots.
    """
    logger.info("Running tuning analysis")

    all_data = [df for _, df in results.items()]
    combined_df = pd.concat(all_data, ignore_index=True)

    plot_path = plots_dir / "tuning_analysis.png"
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    plot_boxplot_by_model(combined_df, "mae", "MAE by Model", "MAE", ax1)
    plot_boxplot_by_model(combined_df, "mape", "MAPE by Model", "MAPE (%)", ax2)
    plot_boxplot_by_model(combined_df, "training_time", "Training Time by Model", "Training Time (s)", ax3)
    plot_mae_convergence(results, ax4)

    fig.suptitle("Tuning Results Overview", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    logger.info("Completed tuning analysis")


def plot_parameters_vs_mae(results: dict[str, pd.DataFrame], plots_dir: Path) -> None:
    """
    Plot scatter plots of parameter vs MAE for each model.

    Args:
        results (dict[str, pd.DataFrame]): Dictionary with model names as keys and DataFrames as values.
        plots_dir (Path): Directory to save the plots.
    """
    for model, df in results.items():
        plot_path = plots_dir / f"{model}_parameters_vs_mae.png"

        param_cols = [
            col
            for col in df.columns
            if col
            not in [
                "model",
                "trial_number",
                "mae",
                "experiment",
                "mape",
                "training_time",
                "prediction_time",
                "evaluation_time",
            ]
        ]

        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        colors = plt.cm.viridis([0.1, 0.5, 0.9])

        for i, param in enumerate(param_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            param_values = df[param]
            mae_values = df["mae"]

            ax.scatter(param_values, mae_values, alpha=0.7, s=35, color=colors[0], edgecolors="white")

            lr = LinearRegression()
            X = param_values.to_numpy().reshape(-1, 1)
            y = mae_values.to_numpy()
            lr.fit(X, y)

            x_trend = np.linspace(param_values.min(), param_values.max(), 100)
            y_trend = lr.predict(x_trend.reshape(-1, 1))
            ax.plot(x_trend, y_trend, "--", color=colors[1], alpha=0.7, linewidth=2)

            top_10_pct = df.nsmallest(max(1, len(df) // 10), "mae")
            top_param_values = top_10_pct[param]
            top_mae_values = top_10_pct["mae"]
            ax.scatter(
                top_param_values,
                top_mae_values,
                color=colors[2],
                s=60,
                alpha=0.7,
                label="Top 10%",
                edgecolors="white",
                linewidth=1,
            )

            ax.legend(loc="upper right", framealpha=0.7)
            ax.set_xlabel(param)
            ax.set_ylabel("MAE")
            ax.set_title(f"{param}", fontweight="bold")
            ax.grid(True, alpha=0.3)

        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        fig.suptitle(f"{model} - Parameters vs MAE", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        logger.info(f"{model} - Parameters vs MAE plot saved to {plot_path}")
