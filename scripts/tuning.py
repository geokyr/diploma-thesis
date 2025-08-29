import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from thesis.common.config import OUTPUTS_DIR, RESULTS_DIRNAME, TUNING_RESULTS_FILENAME
from thesis.common.logger import setup_logger
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.models import ModelType


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


def create_performance_overview(results: Dict[str, pd.DataFrame]):
    """Create overview plots of model performance and timing."""

    # Combine all results for comparison
    all_data = []
    for model_name, df in results.items():
        all_data.append(df)

    if not all_data:
        print("No data available for performance overview")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Tuning Results Overview", fontsize=16)

    # MAE by model
    combined_df.boxplot(column="mae", by="model", ax=axes[0, 0])
    axes[0, 0].set_title("MAE Distribution by Model")
    axes[0, 0].set_xlabel("Model")
    axes[0, 0].set_ylabel("MAE")

    # MAPE by model
    combined_df.boxplot(column="mape", by="model", ax=axes[0, 1])
    axes[0, 1].set_title("MAPE Distribution by Model")
    axes[0, 1].set_xlabel("Model")
    axes[0, 1].set_ylabel("MAPE")

    # Training time by model
    combined_df.boxplot(column="training_time", by="model", ax=axes[1, 0])
    axes[1, 0].set_title("Training Time by Model")
    axes[1, 0].set_xlabel("Model")
    axes[1, 0].set_ylabel("Training Time (s)")

    # Total time by model
    combined_df.boxplot(column="total_time", by="model", ax=axes[1, 1])
    axes[1, 1].set_title("Total Time by Model")
    axes[1, 1].set_xlabel("Model")
    axes[1, 1].set_ylabel("Total Time (s)")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{'=' * 60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 60}")

    for model_name in combined_df["model"].unique():
        model_data = combined_df[combined_df["model"] == model_name]
        best_trial = model_data.loc[model_data["mae"].idxmin()]

        print(f"\n{model_name.upper()}:")
        print(f"  Best MAE: {best_trial['mae']:.3f}")
        print(f"  Best MAPE: {best_trial['mape']:.3f}")
        print(f"  Training time: {best_trial['training_time']:.2f}s")
        print(f"  Total trials: {len(model_data)}")
        print(f"  Mean MAE: {model_data['mae'].mean():.3f} ± {model_data['mae'].std():.3f}")


def create_parameter_vs_mae_plots(results: Dict[str, pd.DataFrame]):
    """Create scatter plots of parameter values vs MAE for each model."""

    for model_name, df in results.items():
        print(f"\nAnalyzing {model_name}...")

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
                "total_time",
            ]
        ]

        if not param_cols:
            print(f"No parameters found for {model_name}")
            continue

        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"{model_name} - Parameter Values vs MAE", fontsize=16)

        for i, param in enumerate(param_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            param_values = df[param]
            mae_values = df["mae"]

            if param_values.dtype in ["int64", "float64"]:
                ax.scatter(param_values, mae_values, alpha=0.6, s=30)

                # Draw trend line
                lr = LinearRegression()
                X = param_values.values.reshape(-1, 1)
                y = mae_values.values
                lr.fit(X, y)

                x_trend = np.linspace(param_values.min(), param_values.max(), 100)
                y_trend = lr.predict(x_trend.reshape(-1, 1))
                ax.plot(x_trend, y_trend, "r--", alpha=0.8, linewidth=2)

                # Calculate correlation
                correlation = param_values.corr(mae_values)

                ax.text(
                    0.05,
                    0.95,
                    f"Corr: {correlation:.3f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

                ax.set_xlabel(param)
                ax.set_ylabel("MAE")

                top_10_pct = df.nsmallest(max(1, len(df) // 10), "mae")
                top_param_values = top_10_pct[param]
                top_mae_values = top_10_pct["mae"]
                ax.scatter(top_param_values, top_mae_values, color="red", s=50, alpha=0.8, label="Top 10%")
                ax.legend()

            else:
                unique_values = param_values.unique()
                mae_by_category = [mae_values[param_values == val].values for val in unique_values]

                box_plot = ax.boxplot(mae_by_category, labels=unique_values, patch_artist=True)

                medians = [np.median(mae_list) for mae_list in mae_by_category]
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(medians)))
                for patch, color in zip(box_plot["boxes"], colors):
                    patch.set_facecolor(color)

                ax.set_xlabel(param)
                ax.set_ylabel("MAE")
                ax.tick_params(axis="x", rotation=45)

            ax.set_title(f"{param}")
            ax.grid(True, alpha=0.3)

        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()


def analyze_parameter_correlations(results: Dict[str, pd.DataFrame]):
    """Analyze correlations between parameters and MAE."""

    print(f"\n{'=' * 60}")
    print("PARAMETER-MAE CORRELATION ANALYSIS")
    print(f"{'=' * 60}")

    for model_name, df in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 40)

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
                "total_time",
            ]
        ]
        numerical_params = [col for col in param_cols if df[col].dtype in ["int64", "float64"]]

        if not numerical_params:
            print("No numerical parameters found")
            continue

        correlations = []
        for param in numerical_params:
            corr = df[param].corr(df["mae"])
            correlations.append((param, corr))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Parameter correlations with MAE:")
        for param, corr in correlations:
            direction = "↑" if corr > 0 else "↓"
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            print(f"  {param:20s}: {corr:6.3f} {direction} ({strength})")

        categorical_params = [col for col in param_cols if df[col].dtype not in ["int64", "float64"]]

        if categorical_params:
            print("\nCategorical parameter analysis:")
            for param in categorical_params:
                mae_by_category = df.groupby(param)["mae"].agg(["mean", "std", "count"])
                best_category = mae_by_category["mean"].idxmin()
                worst_category = mae_by_category["mean"].idxmax()

                print(f"\n  {param}:")
                print(f"    Best category: {best_category} (MAE: {mae_by_category.loc[best_category, 'mean']:.3f})")
                print(f"    Worst category: {worst_category} (MAE: {mae_by_category.loc[worst_category, 'mean']:.3f})")

                for category, stats in mae_by_category.iterrows():
                    print(f"      {category}: {stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']} trials)")


def generate_insights_and_recommendations(results: Dict[str, pd.DataFrame]):
    """Generate insights and recommendations based on parameter-MAE relationships."""

    print(f"\n{'=' * 60}")
    print("INSIGHTS AND RECOMMENDATIONS")
    print(f"{'=' * 60}")

    for model_name, df in results.items():
        print(f"\n{model_name.upper()} RECOMMENDATIONS:")
        print("-" * 40)

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
                "total_time",
            ]
        ]
        numerical_params = [col for col in param_cols if df[col].dtype in ["int64", "float64"]]

        top_trials = df.nsmallest(max(1, len(df) // 10), "mae")

        print(f"Based on top {len(top_trials)} trials (top 10%):")

        for param in numerical_params:
            corr = df[param].corr(df["mae"])

            top_values = top_trials[param]
            all_values = df[param]

            print(f"\n{param}:")
            print(f"  Correlation with MAE: {corr:.3f}")
            print(f"  Top 10% range: {top_values.min():.3f} - {top_values.max():.3f}")
            print(f"  Top 10% mean: {top_values.mean():.3f}")
            print(f"  Overall range: {all_values.min():.3f} - {all_values.max():.3f}")

            if abs(corr) > 0.3:
                if corr < 0:
                    print(f"  💡 INSIGHT: Higher {param} values tend to give better MAE")
                    print("  🎯 RECOMMENDATION: Focus on higher end of range")
                else:
                    print(f"  💡 INSIGHT: Lower {param} values tend to give better MAE")
                    print("  🎯 RECOMMENDATION: Focus on lower end of range")
            else:
                print(f"  💡 INSIGHT: {param} shows weak correlation with MAE")
                print(
                    f"  🎯 RECOMMENDATION: Focus around top 10% range: {top_values.min():.3f} - {top_values.max():.3f}"
                )

        categorical_params = [col for col in param_cols if df[col].dtype not in ["int64", "float64"]]

        for param in categorical_params:
            mae_by_category = df.groupby(param)["mae"].mean()
            best_category = mae_by_category.idxmin()

            print(f"\n{param}:")
            print(
                f"  🎯 RECOMMENDATION: Use '{best_category}' (best average MAE: {mae_by_category[best_category]:.3f})"
            )


def main():
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    results = load_all_tuning_results()

    create_performance_overview(results)
    create_parameter_vs_mae_plots(results)
    analyze_parameter_correlations(results)
    generate_insights_and_recommendations(results)


if __name__ == "__main__":
    main()
