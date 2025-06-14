import matplotlib.pyplot as plt
import pandas as pd

from thesis.common.logger import DATASET_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger
from thesis.dataset.config import PLOTS_DIR

logger = setup_logger(name=DATASET_LOGGER_NAME, log_file=LOG_FILES_CONFIG[DATASET_LOGGER_NAME])


def plot_speed_histogram(speeds_kmh: pd.Series, dataset_id: str, bins: int = 30) -> None:
    """
    Plot a histogram of vehicle speeds in km/h.

    Args:
        speeds_kmh (pd.Series): Series containing vehicle speed data in km/h.
        dataset_id (str): Dataset ID to identify the dataset.
        bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(speeds_kmh, bins=bins)
    plt.title(f"Speed Histogram - {dataset_id}")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Count")
    plt.savefig(PLOTS_DIR / f"{dataset_id}-speed-histogram.png")
    logger.info(f"Saved speed histogram to {PLOTS_DIR / f'{dataset_id}-speed-histogram.png'}")


def plot_average_speed_and_vehicle_count_per_second(
    seconds: pd.Series,
    average_speeds_kmh_per_second: pd.Series,
    vehicle_counts_per_second: pd.Series,
    dataset_id: str,
) -> None:
    """
    Plot average speed and vehicle count on a per-second basis.

    Args:
        seconds (pd.Series): Series containing second-wise data.
        average_speeds_kmh_per_second (pd.Series): Series containing average speed per hour in km/h.
        vehicle_counts_per_second (pd.Series): Series containing vehicle count per second.
        dataset_id (str): Dataset ID to identify the dataset.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(seconds, average_speeds_kmh_per_second, label="Average Speed (km/h)")
    ax1.set_xlabel("Second")
    ax1.set_ylabel("Average Speed (km/h)")

    ax2 = ax1.twinx()
    ax2.plot(seconds, vehicle_counts_per_second, label="Vehicle Count", color="orange")
    ax2.set_ylabel("Count")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper right")

    plt.title(f"Average Speed & Vehicle Count (Per Second) - {dataset_id}")
    plt.savefig(PLOTS_DIR / f"{dataset_id}-average-speed-and-vehicle-count-per-second.png")
    logger.info(
        f"Saved average speed and vehicle count per second to {PLOTS_DIR / f'{dataset_id}-average-speed-and-vehicle-count-per-second.png'}"
    )


def plot_average_speed_and_traffic_generation_period_per_hour(
    hours: pd.Series,
    average_speeds_kmh_per_hour: pd.Series,
    traffic_generation_periods: list[int],
    dataset_id: str,
) -> None:
    """
    Plot average speed and traffic generation period on a per-hour basis.

    Args:
        hours (pd.Series): Series containing hour-wise data.
        average_speeds_kmh_per_hour (pd.Series): Series containing average speed per hour in km/h.
        traffic_generation_periods (list[int]): List of traffic generation periods in seconds.
        dataset_id (str): Dataset ID to identify the dataset.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hours, average_speeds_kmh_per_hour, marker="o", label="Average Speed (km/h)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Speed (km/h)")

    ax2 = ax1.twinx()
    ax2.plot(hours, traffic_generation_periods, marker="o", label="Traffic Generation Period (s)", color="orange")
    ax2.set_ylabel("Traffic Generation Period (s)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc="upper right")

    plt.title(f"Average Speed & Traffic Generation Period (Per Hour) - {dataset_id}")
    plt.savefig(PLOTS_DIR / f"{dataset_id}-average-speed-and-traffic-generation-period-per-hour.png")
    logger.info(
        f"Saved average speed and traffic generation period per hour to {PLOTS_DIR / f'{dataset_id}-average-speed-and-traffic-generation-period-per-hour.png'}"
    )
