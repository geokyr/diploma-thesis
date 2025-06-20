import hashlib
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

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
        logger.info(f"Downloading {filename} from Zenodo...")
        response = requests.get(url, stream=True, headers={"Accept-Encoding": "identity"})
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Successfully downloaded {filename} ({total_size} bytes)")
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
        raise ValueError(f"Unknown dataset file: '{dataset_filename}'")

    expected_md5 = DATASET_FILES_MD5[dataset_filename]

    for attempt in range(max_retries + 1):
        if dataset_path.exists():
            logger.info(f"Verifying integrity of existing file: {dataset_filename}")
            if verify_file_integrity(dataset_path, expected_md5):
                logger.info(f"File {dataset_filename} is available and valid.")
                return
            else:
                logger.warning(f"File {dataset_filename} exists but failed integrity check. Removing corrupted file.")
                dataset_path.unlink()

        logger.info(f"Downloading {dataset_filename}... (attempt {attempt + 1}/{max_retries + 1})")
        success = download_file_from_zenodo(dataset_filename, dataset_path)

        if not success:
            if attempt < max_retries:
                logger.warning("Download failed, retrying...")
                continue
            else:
                raise FileNotFoundError(f"Failed to download {dataset_filename} after {max_retries + 1} attempts.")

        logger.info(f"Verifying integrity of downloaded file: {dataset_filename}")
        if verify_file_integrity(dataset_path, expected_md5):
            logger.info(f"Downloaded file {dataset_filename} is valid.")
            return
        else:
            logger.error(f"Downloaded file {dataset_filename} is invalid.")
            dataset_path.unlink()
            if attempt < max_retries:
                logger.warning("Retrying download...")
                continue
            else:
                raise RuntimeError(f"Downloaded file {dataset_filename} is repeatedly invalid.")

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


def clean_fcd_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean FCD dataset by removing null/nan values and unusable data for training/testing.

    Args:
        df (pd.DataFrame): The FCD dataset to clean

    Returns:
        pd.DataFrame: Cleaned FCD dataset
    """
    logger.info(f"Original dataset shape: {df.shape}")
    df_cleaned = df.dropna()
    logger.info(f"Removed {len(df) - len(df_cleaned)} rows total, new shape: {df_cleaned.shape}")

    return df_cleaned


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
