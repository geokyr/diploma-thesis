"""
High-level pipeline orchestrator for FCD data analysis workflows.
Combines data loading, preprocessing, and exploratory data analysis into streamlined pipeline functions.
"""

import logging
from pathlib import Path

from thesis.common.data import aggregate_fcd_per_hour, generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.eda import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_trip_distance_histogram,
    plot_trip_duration_histogram,
    report_fcd_statistics,
    report_trips_statistics,
)

logger = logging.getLogger(__name__)


def run_fcd_exploratory_data_analysis(
    fcd_csv_path: Path, scenario: str, plots_dir: Path, traffic_generation_periods: list[float]
) -> None:
    """
    Run Exploratory Data Analysis on an FCD dataset.

    Args:
        fcd_csv_path (Path): Path to the FCD CSV file.
        scenario (str): Scenario name.
        plots_dir (Path): Directory to save the generated plots.
        traffic_generation_periods (list[float]): Traffic generation periods.
    """
    logger.info(f"Running FCD Exploratory Data Analysis for {scenario}")

    df_fcd_raw = load_fcd_dataset(fcd_csv_path)
    df_fcd = preprocess_fcd_dataset(df_fcd_raw)
    df_fcd_per_hour = aggregate_fcd_per_hour(df_fcd)

    report_fcd_statistics(df_fcd, scenario)
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_fcd_per_hour, scenario, plots_dir, traffic_generation_periods
    )

    df_trips = generate_trips(df_fcd)

    report_trips_statistics(df_trips, scenario)
    plot_trip_distance_histogram(df_trips, scenario, plots_dir)
    plot_trip_duration_histogram(df_trips, scenario, plots_dir)

    logger.info(f"Completed FCD Exploratory Data Analysis for {scenario}")
