import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from thesis.common.config import (
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TYPE_TEST,
    TYPE_TRAIN,
)
from thesis.eta.data import aggregate_fcd_per_hour

logger = logging.getLogger(__name__)


def report_fcd_statistics(df_fcd: pd.DataFrame, dataset_id: str) -> None:
    """
    Report basic statistics for an FCD DataFrame.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to report statistics for.
        dataset_id (str): Dataset ID to identify the dataset.
    """
    average_speed = df_fcd["vehicle_speed"].mean()
    average_trip_distance = df_fcd.groupby("vehicle_id")["vehicle_odometer"].max().mean()
    unique_vehicles_count = df_fcd["vehicle_id"].nunique()

    logger.info(f"FCD Statistics - {dataset_id}")
    logger.info(f"  Average speed: {average_speed:.2f} km/h")
    logger.info(f"  Average trip distance: {average_trip_distance:.2f} m")
    logger.info(f"  Number of unique vehicles: {unique_vehicles_count}")


def report_trips_statistics(df_trips: pd.DataFrame, dataset_id: str) -> None:
    """
    Report basic statistics for a trips DataFrame.

    Args:
        df_trips (pd.DataFrame): The trips DataFrame to report statistics for.
        dataset_id (str): Dataset ID to identify the dataset.
    """
    logger.info(f"Trips Statistics - {dataset_id}")
    logger.info(f"  Mean distance: {df_trips['distance'].mean():.1f} m")
    logger.info(f"  Median distance: {df_trips['distance'].median():.1f} m")
    logger.info(f"  Mean duration: {df_trips['duration'].mean():.1f} s")
    logger.info(f"  Median duration: {df_trips['duration'].median():.1f} s")


def plot_speed_histogram(df_fcd: pd.DataFrame, dataset_id: str, plots_dir: Path, bins: int = 30) -> None:
    """
    Plot a histogram of vehicle speeds.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to plot the speed histogram for.
        dataset_id (str): Dataset ID to identify the dataset.
        plots_dir (Path): The directory to save the plot to.
        bins (int): Number of bins for the histogram.
    """
    target_dir = plots_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "speed-histogram.png"

    plt.figure(figsize=(6, 4))
    plt.hist(df_fcd["vehicle_speed"], bins=bins)
    plt.title(f"Speed Histogram - {dataset_id}")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Count")
    plt.savefig(target_path)
    plt.close()
    logger.info(f"Speed histogram saved to {target_path}")


def plot_average_speed_and_traffic_generation_period_per_hour(
    df_fcd: pd.DataFrame, dataset_id: str, plots_dir: Path
) -> None:
    """
    Plot average speed and traffic generation period on a per-hour basis.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to plot the average speed and traffic generation period for.
        dataset_id (str): Dataset ID to identify the dataset.
        plots_dir (Path): The directory to save the plot to.

    Raises:
        ValueError: If the dataset ID doesn't contain "train" or "test".
    """
    df_fcd_per_hour = aggregate_fcd_per_hour(df_fcd)
    if TYPE_TRAIN in dataset_id:
        traffic_generation_periods = TRAIN_TRAFFIC_GENERATION_PERIODS
    elif TYPE_TEST in dataset_id:
        traffic_generation_periods = TEST_TRAFFIC_GENERATION_PERIODS
    else:
        raise ValueError(f"Invalid dataset ID, doesn't contain {TYPE_TRAIN} or {TYPE_TEST}: {dataset_id}")

    target_dir = plots_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "average-speed-and-traffic-generation-period-per-hour.png"

    _, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df_fcd_per_hour["hour"], df_fcd_per_hour["average_speed"], marker="o", label="Average Speed (km/h)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Speed (km/h)")

    ax2 = ax1.twinx()
    ax2.plot(
        df_fcd_per_hour["hour"],
        traffic_generation_periods,
        marker="o",
        label="Traffic Generation Period (s)",
        color="orange",
    )
    ax2.set_ylabel("Traffic Generation Period (s)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"Average Speed & Traffic Generation Period (Per Hour) - {dataset_id}")
    plt.savefig(target_path)
    plt.close()
    logger.info(f"Average speed and traffic generation period per hour saved to {target_path}")


def plot_trips_distances_distribution(df_trips: pd.DataFrame, dataset_id: str, plots_dir: Path) -> None:
    """
    Plot the distribution of trips distances.

    Args:
        df_trips (pd.DataFrame): The trips DataFrame to plot the distances distribution for.
        dataset_id (str): Dataset ID to identify the dataset.
        plots_dir (Path): The directory to save the plot to.
    """
    target_dir = plots_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "trips-distances-distribution.png"

    plt.figure(figsize=(6, 4))
    sns.histplot(df_trips["distance"], bins=50, alpha=0.7, edgecolor="black")
    plt.title(f"Distribution of Trips Distances - {dataset_id}")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=df_trips["distance"].mean(),
        linestyle="--",
        label=f"Mean: {df_trips['distance'].mean():.1f} m",
    )
    plt.axvline(
        x=df_trips["distance"].median(),
        linestyle="-",
        label=f"Median: {df_trips['distance'].median():.1f} m",
    )
    plt.legend()
    plt.savefig(target_path)
    plt.close()
    logger.info(f"Trips distances distribution saved to {target_path}")


def plot_trips_durations_distribution(df_trips: pd.DataFrame, dataset_id: str, plots_dir: Path) -> None:
    """
    Plot the distribution of trips durations.

    Args:
        df_trips (pd.DataFrame): The trips DataFrame to plot the durations distribution for.
        dataset_id (str): Dataset ID to identify the dataset.
        plots_dir (Path): The directory to save the plot to.
    """
    target_dir = plots_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "trips-durations-distribution.png"

    plt.figure(figsize=(6, 4))
    sns.histplot(df_trips["duration"], bins=50, alpha=0.7, edgecolor="black")
    plt.title(f"Distribution of Trips Durations - {dataset_id}")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=df_trips["duration"].mean(),
        linestyle="--",
        label=f"Mean: {df_trips['duration'].mean():.1f} s",
    )
    plt.axvline(
        x=df_trips["duration"].median(),
        linestyle="-",
        label=f"Median: {df_trips['duration'].median():.1f} s",
    )
    plt.legend()
    plt.savefig(target_path)
    plt.close()
    logger.info(f"Trips durations distribution saved to {target_path}")
