import logging
from pathlib import Path

import pandas as pd

from thesis.common.config import END_TIME, MIN_DISTANCE, MIN_DURATION

logger = logging.getLogger(__name__)


def load_fcd_dataset(fcd_csv_path: Path) -> pd.DataFrame:
    """
    Load the FCD dataset from a CSV file.

    Args:
        fcd_csv_path (Path): Path to the FCD data file.

    Returns:
        pd.DataFrame: DataFrame containing the raw FCD data.
    """
    logger.info(f"Loading FCD dataset from {fcd_csv_path}")

    dtype = {
        "timestep_time": int,
        "vehicle_acceleration": float,
        "vehicle_id": str,
        "vehicle_odometer": float,
        "vehicle_speed": float,
        "vehicle_x": float,
        "vehicle_y": float,
    }
    df_fcd_raw = pd.read_csv(fcd_csv_path, sep=";", header=0, dtype=dtype)

    logger.info(f"Loaded {len(df_fcd_raw)} rows of FCD data")
    return df_fcd_raw


def preprocess_fcd_dataset(df_fcd_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess FCD DataFrame by removing nulls, filtering to END_TIME, and converting speed to km/h.

    Args:
        df_fcd_raw (pd.DataFrame): Raw FCD DataFrame.

    Returns:
        pd.DataFrame: Preprocessed FCD DataFrame.
    """
    logger.info(f"Preprocessing FCD dataset, initial shape {df_fcd_raw.shape}")
    df_fcd = df_fcd_raw.dropna().reset_index(drop=True)
    df_fcd = df_fcd[df_fcd["timestep_time"] < END_TIME]
    df_fcd["vehicle_speed"] = df_fcd["vehicle_speed"] * 3.6

    logger.info(f"Completed FCD preprocessing, final shape {df_fcd.shape}")
    return df_fcd


def aggregate_fcd_per_hour(df_fcd: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an FCD DataFrame per hour, producing summaries of average speed and vehicle count.

    Args:
        df_fcd (pd.DataFrame): FCD DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing average speed and vehicle count per hour.
    """
    logger.info("Aggregating FCD per hour")
    df_fcd_with_hour = df_fcd.assign(hour=(df_fcd["timestep_time"] // 3600))
    df_fcd_per_hour = (
        df_fcd_with_hour.groupby("hour")
        .agg(average_speed=("vehicle_speed", "mean"), vehicle_count=("vehicle_id", "nunique"))
        .reset_index()
    )

    logger.info(f"Completed FCD aggregation for {len(df_fcd_per_hour)} hours")
    return df_fcd_per_hour


def generate_trips(
    df_fcd: pd.DataFrame, min_duration: int = MIN_DURATION, min_distance: int = MIN_DISTANCE
) -> pd.DataFrame:
    """
    Generate trips from the FCD DataFrame, filtering out short trips.

    Args:
        df_fcd (pd.DataFrame): FCD DataFrame.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.

    Returns:
        pd.DataFrame: A DataFrame containing the trips.
    """
    logger.info("Generating trips")
    df_sorted = df_fcd.sort_values("timestep_time")

    df_trips = (
        df_sorted.groupby("vehicle_id")
        .agg(
            source_x=("vehicle_x", "first"),
            source_y=("vehicle_y", "first"),
            destination_x=("vehicle_x", "last"),
            destination_y=("vehicle_y", "last"),
            odometer_start=("vehicle_odometer", "first"),
            odometer_end=("vehicle_odometer", "last"),
            time_start=("timestep_time", "first"),
            time_end=("timestep_time", "last"),
        )
        .assign(
            duration=lambda d: d.time_end - d.time_start,
            distance=lambda d: d.odometer_end - d.odometer_start,
        )
        .query(f"duration >= {min_duration} and distance >= {min_distance}")
        .loc[
            :,
            [
                "source_x",
                "source_y",
                "destination_x",
                "destination_y",
                "time_start",
                "distance",
                "duration",
            ],
        ]
        .reset_index(drop=True)
    )

    logger.info(f"Generated {len(df_trips)} trips")
    return df_trips
