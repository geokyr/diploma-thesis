import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from thesis.common.config import RANDOM_SEED
from thesis.eta.config import DATASET_FILES_MD5, ZENODO_BASE_URL

logger = logging.getLogger(__name__)


def calculate_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path (Path): The path to the file to calculate the MD5 hash of.
        chunk_size (int): The size of the chunks to read from the file.

    Returns:
        str: The MD5 hash of the file.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_file_integrity(file_path: Path, expected_md5: str) -> bool:
    """
    Verify file integrity using MD5 checksum.

    Args:
        file_path (Path): The path to the file to verify the integrity of.
        expected_md5 (str): The expected MD5 hash of the file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    actual_md5 = calculate_md5(file_path)
    return actual_md5 == expected_md5


def download_file_from_zenodo(filename: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a single file from Zenodo with progress bar.

    Args:
        filename (str): The name of the file to download.
        output_path (Path): The path to save the downloaded file.
        chunk_size (int): The size of the chunks to read from the file.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.
    """
    url = f"{ZENODO_BASE_URL}/files/{filename}"

    try:
        response = requests.get(url, stream=True, headers={"Accept-Encoding": "identity"})
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        logger.info(f"Downloading {filename} from Zenodo ({total_size} bytes)")

        with open(output_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download {filename}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def ensure_dataset_is_available_and_valid(dataset_path: Path, max_retries: int = 1) -> None:
    """
    Ensure that a dataset file is available and valid, downloading it from Zenodo if missing.

    Args:
        dataset_path (Path): The path where the dataset file should be located.
        max_retries (int): Maximum number of download/verification retries.

    Raises:
        ValueError: If the filename is not in the known dataset files.
        FileNotFoundError: If the file cannot be downloaded or verified after retries.
        RuntimeError: If file is repeatedly invalid.
    """
    dataset_filename = dataset_path.name

    if dataset_filename not in DATASET_FILES_MD5:
        raise ValueError(f"Unknown dataset file: {dataset_filename}")

    expected_md5 = DATASET_FILES_MD5[dataset_filename]

    for attempt in range(max_retries + 1):
        if dataset_path.exists():
            if verify_file_integrity(dataset_path, expected_md5):
                logger.info(f"File {dataset_filename} is available and valid")
                return
            else:
                logger.warning(f"Removing existing file {dataset_filename} after failed integrity check")
                dataset_path.unlink()

        logger.info(f"Downloading {dataset_filename} (attempt {attempt + 1}/{max_retries + 1})")
        success = download_file_from_zenodo(dataset_filename, dataset_path)

        if not success:
            if attempt < max_retries:
                logger.warning("Retrying failed download")
                continue
            else:
                raise FileNotFoundError(f"Failed to download {dataset_filename} after {max_retries + 1} attempts")

        if verify_file_integrity(dataset_path, expected_md5):
            logger.info(f"Downloaded file {dataset_filename} is valid")
            return
        else:
            logger.error(f"Downloaded file {dataset_filename} is invalid")
            dataset_path.unlink()
            if attempt < max_retries:
                logger.warning("Retrying failed download")
                continue
            else:
                raise RuntimeError(f"Downloaded file {dataset_filename} is repeatedly invalid")

    raise RuntimeError(f"Unexpected error while ensuring availability and validity of {dataset_filename}")


def load_fcd_dataset(fcd_path: Path) -> pd.DataFrame:
    """
    Load the FCD dataset from a file, downloading it from Zenodo if missing.

    Args:
        fcd_path (Path): The path to the FCD data file.

    Returns:
        pd.DataFrame: A DataFrame containing the FCD data.
    """
    ensure_dataset_is_available_and_valid(fcd_path)

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

    logger.info(f"Loaded {len(df)} rows of FCD data from {fcd_path}")
    return df


def preprocess_fcd_dataset(df_fcd: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess FCD DataFrame by removing nulls, filtering to 10 hours, and converting speed to km/h.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed FCD DataFrame.
    """
    initial_shape = df_fcd.shape
    df_fcd = df_fcd.dropna().reset_index(drop=True)
    df_fcd = df_fcd[df_fcd["timestep_time"] < 36000]
    df_fcd["vehicle_speed"] = df_fcd["vehicle_speed"] * 3.6
    final_shape = df_fcd.shape

    logger.info(f"Completed FCD preprocessing, with initial shape {initial_shape} and final shape {final_shape}")

    return df_fcd


def aggregate_fcd_per_hour(df_fcd: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate an FCD DataFrame per hour, producing summaries of average speed and vehicle count.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to aggregate.

    Returns:
        pd.DataFrame: A DataFrame containing average speed and vehicle count per hour.
    """
    df_with_hour = df_fcd.assign(hour=(df_fcd["timestep_time"] // 3600))
    df_per_hour = (
        df_with_hour.groupby("hour")
        .agg(average_speed=("vehicle_speed", "mean"), vehicle_count=("vehicle_id", "nunique"))
        .reset_index()
    )

    logger.info(f"Completed FCD aggregation for {len(df_per_hour)} hours")
    return df_per_hour


def generate_full_trips(df_fcd: pd.DataFrame, min_duration: int = 60, min_distance: int = 500) -> pd.DataFrame:
    """
    Generate the full trips from the FCD DataFrame, filtering out short trips.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to generate full trips from.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.

    Returns:
        pd.DataFrame: A DataFrame containing the full trips.
    """
    df_sorted = df_fcd.sort_values("timestep_time")

    df_full_trips = (
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
            hour_bin=lambda d: d.time_start // 3600,
        )
        .query(f"duration >= {min_duration} and distance >= {min_distance}")
        .loc[:, ["source_x", "source_y", "destination_x", "destination_y", "hour_bin", "distance", "duration"]]
        .reset_index(drop=True)
    )

    return df_full_trips


def generate_sub_trips(
    group: pd.DataFrame,
    augmentation_rate: float,
    min_trip_ratio: float,
    min_duration: int,
    min_distance: int,
) -> pd.DataFrame:
    """
    Generate sub-trips from a vehicle's FCD group, filtering out short trips.

    Args:
        group (pd.DataFrame): The FCD DataFrame to generate sub-trips from.
        augmentation_rate (float): The rate at which to augment the trips.
        min_trip_ratio (float): The minimum ratio of the trip to be augmented.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.

    Returns:
        pd.DataFrame: A DataFrame containing the sub-trips for the vehicle.
    """
    timesteps = group["timestep_time"].to_numpy()
    x_positions = group["vehicle_x"].to_numpy()
    y_positions = group["vehicle_y"].to_numpy()
    odometer_readings = group["vehicle_odometer"].to_numpy()

    n = len(group)
    n_sub_trips = int(n * augmentation_rate)
    minimum_length = max(2, int(n * min_trip_ratio))
    rng = np.random.default_rng(RANDOM_SEED)

    sub_trips_records = []
    for _ in range(n_sub_trips):
        start_idx = rng.integers(0, n - minimum_length + 1)
        sub_trip_length = rng.integers(minimum_length, n - start_idx + 1)
        end_idx = start_idx + sub_trip_length - 1

        sub_trip_duration = timesteps[end_idx] - timesteps[start_idx]
        sub_trip_distance = odometer_readings[end_idx] - odometer_readings[start_idx]
        if sub_trip_duration < min_duration or sub_trip_distance < min_distance:
            continue

        sub_trips_records.append(
            {
                "source_x": x_positions[start_idx],
                "source_y": y_positions[start_idx],
                "destination_x": x_positions[end_idx],
                "destination_y": y_positions[end_idx],
                "hour_bin": timesteps[start_idx] // 3600,
                "distance": sub_trip_distance,
                "duration": sub_trip_duration,
            }
        )

    return pd.DataFrame.from_records(sub_trips_records)


def generate_trips(
    df_fcd: pd.DataFrame,
    min_duration: int = 60,
    min_distance: int = 500,
    augmentation_rate: float = 0.0,
    min_trip_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    Generate trips with optional sub-trip augmentation.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to generate trips from.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.
        augmentation_rate (float): Multiplier for sub-trip augmentation based on each trip's length.
        min_trip_ratio (float): The minimum ratio of the trip to be augmented.

    Returns:
        pd.DataFrame: A DataFrame containing the generated trips.
    """
    df_full_trips = generate_full_trips(df_fcd, min_duration, min_distance)
    logger.info(f"Generated {len(df_full_trips)} full trips")

    if augmentation_rate <= 0 or df_full_trips.empty:
        return df_full_trips

    sub_trips_list = []
    for _, group in df_fcd.sort_values("timestep_time").groupby("vehicle_id"):
        sub_trips = generate_sub_trips(group, augmentation_rate, min_trip_ratio, min_duration, min_distance)
        sub_trips_list.append(sub_trips)

    df_sub_trips = pd.concat(sub_trips_list, ignore_index=True)
    logger.info(f"Generated {len(df_sub_trips)} sub-trips")

    df_trips = pd.concat([df_full_trips, df_sub_trips], ignore_index=True)
    logger.info(f"Generated {len(df_trips)} trips")

    return df_trips
