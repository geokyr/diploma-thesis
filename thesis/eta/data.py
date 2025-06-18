from pathlib import Path

import pandas as pd

from thesis.logger import ETA_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME, log_file=LOG_FILES_CONFIG[ETA_LOGGER_NAME])


def load_fcd_dataset(fcd_path: Path) -> pd.DataFrame:
    """
    Load the FCD dataset from a file.

    Args:
        fcd_path (Path): The path to the FCD data file.

    Returns:
        pd.DataFrame: A DataFrame containing the FCD data.
    """
    logger.info(f"Loading FCD data from {fcd_path}...")

    dtype = {
        "timestep_time": int,
        "vehicle_acceleration": float,
        "vehicle_id": str,
        "vehicle_odometer": float,
        "vehicle_speed": float,
        "vehicle_x": float,
        "vehicle_y": float,
    }
    df = pd.read_csv(fcd_path, sep=";", header=0, dtype=dtype)

    logger.info(f"Loaded {len(df)} rows of FCD data.")
    return df


def prepare_baseline_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the baseline trips from the FCD data.

    Args:
        df (pd.DataFrame): The FCD data.

    Returns:
        pd.DataFrame: A DataFrame containing the baseline trips.
    """
    logger.info("Preparing baseline trips...")

    trips = []
    for _, group in df.groupby("vehicle_id"):
        start = group.iloc[0]
        end = group.iloc[-1]
        if end["timestep_time"] - start["timestep_time"] <= 0:
            continue
        trips.append(
            {
                "origin_x": start["vehicle_x"],
                "origin_y": start["vehicle_y"],
                "destination_x": end["vehicle_x"],
                "destination_y": end["vehicle_y"],
                "hour_bin": start["timestep_time"] // 3600,
                "distance": end["vehicle_odometer"] - start["vehicle_odometer"],
                "duration": end["timestep_time"] - start["timestep_time"],
            }
        )
    trips_df = pd.DataFrame(trips)

    logger.info(f"Prepared {len(trips_df)} baseline trips.")
    return trips_df
