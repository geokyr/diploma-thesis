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
    fcd_csv_path: Path, id_df: str, plots_dir: Path, traffic_generation_periods: tuple[float, ...]
) -> None:
    """
    Run Exploratory Data Analysis on an FCD dataset.

    Args:
        fcd_csv_path (Path): Path to the FCD CSV file.
        id_df (str): Identifier for the DataFrame.
        plots_dir (Path): Directory to save the generated plots.
        traffic_generation_periods (tuple[float, ...]): Traffic generation periods.
    """
    logger.info(f"Running FCD Exploratory Data Analysis for {id_df}")

    df_fcd = load_fcd_dataset(fcd_csv_path)
    df_fcd = preprocess_fcd_dataset(df_fcd)
    df_fcd_per_hour = aggregate_fcd_per_hour(df_fcd)

    report_fcd_statistics(df_fcd, id_df)
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_fcd_per_hour, id_df, plots_dir, traffic_generation_periods
    )

    df_trips = generate_trips(df_fcd)

    report_trips_statistics(df_trips, id_df)
    plot_trip_distance_histogram(df_trips, id_df, plots_dir)
    plot_trip_duration_histogram(df_trips, id_df, plots_dir)

    logger.info(f"Completed FCD Exploratory Data Analysis for {id_df}")
