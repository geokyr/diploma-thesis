import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from thesis.common.config import (
    AUGMENTATION_RATE,
    MIN_DISTANCE,
    MIN_DURATION,
    MIN_TRIP_RATIO,
    RANDOM_SEED_DEFAULT,
    ZENODO_DATASET_API_URL,
)
from thesis.common.data import generate_trips

logger = logging.getLogger(__name__)


def _verify_file_integrity(file_path: Path, expected_md5: str, chunk_size: int = 8192) -> bool:
    """
    Verify file integrity using MD5 checksum.

    Args:
        file_path (Path): The path to the file to verify the integrity of.
        expected_md5 (str): The expected MD5 hash of the file.
        chunk_size (int): The size of the chunks to read from the file.

    Returns:
        bool: True if the file is valid, False otherwise.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    file_md5 = hash_md5.hexdigest()

    return file_md5 == expected_md5


def _get_file_md5_from_zenodo(filename: str) -> str:
    """
    Fetch MD5 hash for a specific file from Zenodo API.

    Args:
        filename (str): The name of the file to get MD5 hash for

    Returns:
        str: The MD5 hash

    Raises:
        requests.RequestException: If the API request fails
        ValueError: If the checksum format is unexpected
    """
    file_url = f"{ZENODO_DATASET_API_URL}/files/{filename}"
    response = requests.get(file_url)
    response.raise_for_status()
    file_data = response.json()
    checksum = file_data.get("checksum")

    if checksum and checksum.startswith("md5:"):
        md5_hash = checksum[4:]
        return md5_hash
    else:
        error_msg = f"Unexpected checksum format for {filename}: {checksum}"
        logger.error(error_msg)
        raise ValueError(error_msg)


def _download_file_from_zenodo(filename: str, output_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a single file from Zenodo with progress bar.

    Args:
        filename (str): The name of the file to download.
        output_path (Path): The path to save the downloaded file.
        chunk_size (int): The size of the chunks to read from the file.

    Raises:
        requests.RequestException: If the file cannot be downloaded.
    """
    try:
        download_url = f"{ZENODO_DATASET_API_URL}/files/{filename}/content"
        response = requests.get(download_url, stream=True, headers={"Accept-Encoding": "identity"})
        response.raise_for_status()

        file_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f, tqdm(total=file_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    except requests.RequestException:
        output_path.unlink(missing_ok=True)
        raise


def ensure_dataset_is_valid(dataset_path: Path) -> None:
    """
    Ensure that a dataset file is valid, downloading it from Zenodo if missing.

    Args:
        dataset_path (Path): The path where the dataset file should be located.

    Raises:
        FileNotFoundError: If the file doesn't exist and cannot be downloaded or verified
        requests.RequestException: If downloading or fetching MD5 from Zenodo fails
        ValueError: If the MD5 checksum format from Zenodo is unexpected
    """
    dataset_filename = dataset_path.name

    if not dataset_path.exists():
        logger.info(f"File {dataset_filename} not found, downloading from Zenodo")
        try:
            _download_file_from_zenodo(dataset_filename, dataset_path)
        except requests.RequestException as e:
            error_msg = f"Failed to download {dataset_filename}: {e}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e

    logger.info(f"Verifying integrity of file {dataset_filename}")
    try:
        expected_md5 = _get_file_md5_from_zenodo(dataset_filename)
    except requests.RequestException as e:
        error_msg = f"Failed to fetch MD5 hash for {dataset_filename}: {e}"
        logger.error(error_msg)
        raise
    except ValueError as e:
        logger.error(f"Invalid MD5 format for {dataset_filename}: {e}")
        raise

    if _verify_file_integrity(dataset_path, expected_md5):
        logger.info(f"File {dataset_filename} is valid")
    else:
        error_msg = f"File {dataset_filename} failed integrity check"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def _generate_sub_trips(
    group: pd.DataFrame,
    augmentation_rate: float,
    min_trip_ratio: float,
    min_duration: int,
    min_distance: int,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Generate sub-trips from a vehicle's FCD group, filtering out short trips.

    Args:
        group (pd.DataFrame): The FCD DataFrame to generate sub-trips from.
        augmentation_rate (float): The rate at which to augment the trips.
        min_trip_ratio (float): The minimum ratio of the trip to be augmented.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.
        random_seed (int): The random seed to use for the random number generator.

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
    rng = np.random.default_rng(random_seed)

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
                "time_start": timesteps[start_idx],
                "distance": sub_trip_distance,
                "duration": sub_trip_duration,
            }
        )

    return pd.DataFrame.from_records(sub_trips_records)


def generate_full_trips(
    df_fcd: pd.DataFrame,
    min_duration: int = MIN_DURATION,
    min_distance: int = MIN_DISTANCE,
    augmentation_rate: float = AUGMENTATION_RATE,
    min_trip_ratio: float = MIN_TRIP_RATIO,
) -> pd.DataFrame:
    """
    Generate full trips with optional sub-trip augmentation.

    Args:
        df_fcd (pd.DataFrame): The FCD DataFrame to generate full trips from.
        min_duration (int): Minimum trip duration in seconds.
        min_distance (int): Minimum trip distance in meters.
        augmentation_rate (float): Multiplier for sub-trip augmentation based on each trip's length.
        min_trip_ratio (float): The minimum ratio of the trip to be augmented.

    Returns:
        pd.DataFrame: A DataFrame containing the generated full trips.
    """
    df_trips = generate_trips(df_fcd, min_duration, min_distance)

    if augmentation_rate <= 0 or df_trips.empty:
        return df_trips

    logger.info("Generating sub-trips")
    sub_trips_list = []
    for _, group in df_fcd.sort_values("timestep_time").groupby("vehicle_id"):
        sub_trips = _generate_sub_trips(group, augmentation_rate, min_trip_ratio, min_duration, min_distance)
        sub_trips_list.append(sub_trips)

    df_sub_trips = pd.concat(sub_trips_list, ignore_index=True)
    logger.info(f"Generated {len(df_sub_trips)} sub-trips")

    df_full_trips = pd.concat([df_trips, df_sub_trips], ignore_index=True)
    logger.info(f"Generated {len(df_full_trips)} full trips")

    return df_full_trips
