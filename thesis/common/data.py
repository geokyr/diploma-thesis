"""
Core data processing utilities for FCD (Floating Car Data) and trip generation.
Provides functions for loading, preprocessing, and transforming simulation data into analysis-ready formats.
"""

import logging
from pathlib import Path

import pandas as pd

from thesis.common.config import END_TIME, MIN_DISTANCE, MIN_DURATION, NUM_RETRAINING_TRIPS

logger = logging.getLogger(__name__)


def load_fcd_dataset(fcd_parquet_path: Path) -> pd.DataFrame:
    """
    Load the FCD dataset from a Parquet file.

    Args:
        fcd_parquet_path (Path): Path to the FCD Parquet data file.

    Returns:
        pd.DataFrame: DataFrame containing the raw FCD data.
    """
    logger.info(f"Loading FCD dataset from {fcd_parquet_path}")

    df_fcd_raw = pd.read_parquet(fcd_parquet_path)

    logger.info(f"Loaded {len(df_fcd_raw)} rows of FCD data")

    return df_fcd_raw


def load_fcd_dataset_csv(fcd_csv_path: Path) -> pd.DataFrame:
    """
    Load the FCD dataset from a CSV file.

    Args:
        fcd_csv_path (Path): Path to the FCD CSV data file.

    Returns:
        pd.DataFrame: DataFrame containing the raw FCD data.
    """
    logger.info(f"Loading FCD dataset from {fcd_csv_path}")

    dtype = {
        "timestep_time": int,
        "vehicle_fuel": float,
        "vehicle_id": str,
        "vehicle_lane": str,
        "vehicle_odometer": float,
        "vehicle_speed": float,
        "vehicle_waiting": float,
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


def get_adaptation_test_data(trips_rain: pd.DataFrame, n_rain_trips: int = NUM_RETRAINING_TRIPS) -> pd.DataFrame:
    """
    Get adaptation test data, by removing data that will be used for retraining.

    Args:
        trips_rain (pd.DataFrame): Dataframe with trips from rainy weather
        n_rain_trips (int): Number of rain trips used for retraining

    Returns:
        pd.DataFrame: Dataframe with adaptation test data
    """
    return trips_rain.sort_values("time_start").iloc[n_rain_trips:].reset_index(drop=True)


def get_adaptation_retrain_data(
    trips_test: pd.DataFrame,
    trips_rain: pd.DataFrame,
    n_test_trips: int = NUM_RETRAINING_TRIPS,
    n_rain_trips: int = NUM_RETRAINING_TRIPS,
) -> pd.DataFrame:
    """
    Get adaptation retrain data, by combining test and rain data.

    Args:
        trips_test (pd.DataFrame): Test dataset trips
        trips_rain (pd.DataFrame): Rain dataset trips
        n_test_trips (int): Number of test trips used for retrain
        n_rain_trips (int): Number of raintrips used for retrain

    Returns:
        pd.DataFrame: Dataframe with adaptation retrain data
    """
    retrain_test_data = trips_test.nlargest(n_test_trips, "time_start")
    retrain_rain_data = trips_rain.nsmallest(n_rain_trips, "time_start")

    retrain_data = pd.concat([retrain_test_data, retrain_rain_data], ignore_index=True)

    return retrain_data
