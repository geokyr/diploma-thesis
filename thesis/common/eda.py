"""Exploratory Data Analysis utilities for traffic simulation data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def report_fcd_statistics(df_fcd: pd.DataFrame, scenario: str) -> None:
    """
    Report basic statistics for an FCD DataFrame.

    Args:
        df_fcd (pd.DataFrame): FCD DataFrame to report statistics for.
        scenario (str): Scenario name.
    """
    unique_vehicles_count = df_fcd["vehicle_id"].nunique()
    average_trip_distance = df_fcd.groupby("vehicle_id")["vehicle_odometer"].max().mean()
    average_speed = df_fcd["vehicle_speed"].mean()

    logger.info(f"FCD Statistics - {scenario}")
    logger.info(f"Number of unique vehicles: {unique_vehicles_count}")
    logger.info(f"Average trip distance: {average_trip_distance:.2f} m")
    logger.info(f"Average speed: {average_speed:.2f} km/h")


def plot_average_speed_and_traffic_generation_period_per_hour(
    df_fcd_per_hour: pd.DataFrame, scenario: str, plots_dir: Path, traffic_generation_periods: list[float]
) -> None:
    """
    Plot average speed and traffic generation period on a per-hour basis.

    Args:
        df_fcd_per_hour (pd.DataFrame): FCD DataFrame aggregated per hour to plot the average speed and traffic generation period for.
        scenario (str): Scenario name.
        plots_dir (Path): Directory to save the plot to.
        traffic_generation_periods (list[float]): Traffic generation periods.
    """
    plot_path = plots_dir / f"{scenario}_average_speed_and_traffic_generation_period_per_hour.png"

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
        color="tab:orange",
    )
    ax2.set_ylabel("Traffic Generation Period (s)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(f"Average Speed & Traffic Generation Period (Per Hour) - {scenario}")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Average speed and traffic generation period per hour saved to {plot_path}")


def report_trips_statistics(df_trips: pd.DataFrame, scenario: str) -> None:
    """
    Report basic statistics for a trips DataFrame.

    Args:
        df_trips (pd.DataFrame): Trips DataFrame to report statistics for.
        scenario (str): Scenario name.
    """
    average_distance = df_trips["distance"].mean()
    average_duration = df_trips["duration"].mean()
    total_trips = len(df_trips)

    logger.info(f"Trips Statistics - {scenario}")
    logger.info(f"Total number of trips: {total_trips}")
    logger.info(f"Average trip distance: {average_distance:.2f} m")
    logger.info(f"Average trip duration: {average_duration:.2f} s")


def plot_trip_distance_histogram(df_trips: pd.DataFrame, scenario: str, plots_dir: Path, bins: int = 50) -> None:
    """
    Plot a histogram of trip distances.

    Args:
        df_trips (pd.DataFrame): Trips DataFrame to plot the distance histogram for.
        scenario (str): Scenario name.
        plots_dir (Path): Directory to save the plot to.
        bins (int): Number of bins for the histogram.
    """
    plot_path = plots_dir / f"{scenario}_trip_distance_histogram.png"

    plt.figure(figsize=(6, 4))
    plt.hist(df_trips["distance"], bins=bins, edgecolor="black")
    plt.title(f"Trip Distance Histogram - {scenario}")
    plt.xlabel("Distance (m)")
    plt.ylabel("Count")
    plt.axvline(
        x=df_trips["distance"].mean(),
        linestyle="--",
        color="tab:orange",
        label=f"Mean: {df_trips['distance'].mean():.1f} m",
    )
    plt.axvline(
        x=df_trips["distance"].median(),
        linestyle="-",
        color="tab:orange",
        label=f"Median: {df_trips['distance'].median():.1f} m",
    )
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Trip distance histogram saved to {plot_path}")


def plot_trip_duration_histogram(df_trips: pd.DataFrame, scenario: str, plots_dir: Path, bins: int = 50) -> None:
    """
    Plot a histogram of trip durations.

    Args:
        df_trips (pd.DataFrame): Trips DataFrame to plot the duration histogram for.
        scenario (str): Scenario name.
        plots_dir (Path): Directory to save the plot to.
        bins (int): Number of bins for the histogram.
    """
    plot_path = plots_dir / f"{scenario}_trip_duration_histogram.png"

    plt.figure(figsize=(6, 4))
    plt.hist(df_trips["duration"], bins=bins, edgecolor="black")
    plt.title(f"Trip Duration Histogram - {scenario}")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.axvline(
        x=df_trips["duration"].mean(),
        linestyle="--",
        color="tab:orange",
        label=f"Mean: {df_trips['duration'].mean():.1f} s",
    )
    plt.axvline(
        x=df_trips["duration"].median(),
        linestyle="-",
        color="tab:orange",
        label=f"Median: {df_trips['duration'].median():.1f} s",
    )
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Trip duration histogram saved to {plot_path}")
