import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def report_trip_statistics(trips_distances: pd.Series, trips_durations: pd.Series) -> None:
    """
    Report basic statistics about trip distances and durations.

    Args:
        trips_distances (pd.Series): The trip distances.
        trips_durations (pd.Series): The trip durations.
    """
    logger.info("Trip Distances Statistics")
    logger.info(f"  Mean: {trips_distances.mean():.1f} m")
    logger.info(f"  Median: {trips_distances.median():.1f} m")
    logger.info(f"  Std: {trips_distances.std():.1f} m")
    logger.info(f"  Min: {trips_distances.min():.1f} m")
    logger.info(f"  Max: {trips_distances.max():.1f} m")

    logger.info("Trip Durations Statistics")
    logger.info(f"  Mean: {trips_durations.mean():.1f} s")
    logger.info(f"  Median: {trips_durations.median():.1f} s")
    logger.info(f"  Std: {trips_durations.std():.1f} s")
    logger.info(f"  Min: {trips_durations.min():.1f} s")
    logger.info(f"  Max: {trips_durations.max():.1f} s")


def plot_trip_distances_distribution(trips_distances: pd.Series) -> None:
    """
    Plot the distribution of trip distances.

    Args:
        trips_distances (pd.Series): The trip distances.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(trips_distances, bins=50, alpha=0.7, edgecolor="black")
    plt.title("Distribution of Trip Distances")
    plt.xlabel("Distance (meters)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=trips_distances.mean(),
        linestyle="--",
        label=f"Mean: {trips_distances.mean():.1f} m",
    )
    plt.axvline(
        x=trips_distances.median(),
        linestyle="-",
        label=f"Median: {trips_distances.median():.1f} m",
    )
    plt.legend()
    plt.show()


def plot_trip_durations_distribution(trips_durations: pd.Series) -> None:
    """
    Plot the distribution of trip durations.

    Args:
        trips_durations (pd.Series): The trip durations.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(trips_durations, bins=50, alpha=0.7, edgecolor="black")
    plt.title("Distribution of Trip Durations")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Frequency")
    plt.axvline(
        x=trips_durations.mean(),
        linestyle="--",
        label=f"Mean: {trips_durations.mean():.1f} s",
    )
    plt.axvline(
        x=trips_durations.median(),
        linestyle="-",
        label=f"Median: {trips_durations.median():.1f} s",
    )
    plt.legend()
    plt.show()
