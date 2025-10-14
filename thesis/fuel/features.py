"""Feature engineering and transformation utilities for fuel consumption prediction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from thesis.common.config import (
    FEATURE_CALIBRATOR_FILENAME,
    MIN_TRIP_POINTS,
    N_END_CLUSTERS,
    N_START_CLUSTERS,
    RANDOM_SEED_DEFAULT,
    START_HOUR_MAX,
)

CORE_FEATURES = [
    "timestep_time_min",
    "start_hour",
    "vehicle_x_first",
    "vehicle_y_first",
    "vehicle_x_last",
    "vehicle_y_last",
    "trip_actual_distance",
    "spatial_extent",
    "straight_line_distance",
]

logger = logging.getLogger(__name__)


def check_required_columns(df: pd.DataFrame, required_columns: list[str], feature_type: str) -> None:
    """
    Check if DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): DataFrame to check.
        required_columns (list[str]): List of required column names.
        feature_type (str): Type of features being added (for logging).

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]

        error_msg = f"{missing_columns} not found, while adding {feature_type} features"
        logger.error(error_msg)
        raise ValueError(error_msg)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to FCD data.

    Args:
        df (pd.DataFrame): Raw FCD DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added time features.
    """
    df = df.sort_values(["vehicle_id", "timestep_time"])
    df["hour_bin"] = (df["timestep_time"] // 3600).astype(int)
    return df.fillna(0)


def create_trips_simple(df: pd.DataFrame, min_trip_points: int = MIN_TRIP_POINTS) -> pd.DataFrame:
    """
    Create trips from FCD data: one vehicle_id = one trip.
    Keeps only vehicles with at least `min_trip_points` rows.

    Args:
        df (pd.DataFrame): FCD DataFrame with time features.
        min_trip_points (int): Minimum number of points required per trip.

    Returns:
        pd.DataFrame: DataFrame with vehicle_id column added.
    """
    df = df.sort_values(["vehicle_id", "timestep_time"])
    df = df.drop_duplicates(subset=["vehicle_id", "timestep_time"])

    counts = df.groupby("vehicle_id").size()
    valid_ids = counts[counts >= min_trip_points].index

    df_out = df[df["vehicle_id"].isin(valid_ids)].copy()

    return df_out


def create_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trip-level features for fuel consumption from FCD data.

    Args:
        df (pd.DataFrame): FCD DataFrame with vehicle_id column.

    Returns:
        pd.DataFrame: DataFrame with trip-level features.
    """
    trip_features = df.groupby("vehicle_id").agg(
        {
            "timestep_time": ["min"],
            "vehicle_x": ["first", "last", "min", "max"],
            "vehicle_y": ["first", "last", "min", "max"],
            "vehicle_odometer": ["min", "max"],
            "hour_bin": ["first"],
            "vehicle_fuel": ["sum"],
        }
    )

    trip_features.columns = ["_".join(col).strip() for col in trip_features.columns.values]

    if "vehicle_fuel_sum" in trip_features.columns:
        trip_features["vehicle_fuel_log_sum"] = np.log1p(trip_features["vehicle_fuel_sum"])

    trip_features.drop(columns=["vehicle_fuel_sum"], inplace=True)

    trip_features["trip_actual_distance"] = (
        trip_features["vehicle_odometer_max"] - trip_features["vehicle_odometer_min"]
    )
    trip_features.drop(columns=["vehicle_odometer_max", "vehicle_odometer_min"], inplace=True)
    trip_features["start_hour"] = trip_features["hour_bin_first"] % 24
    trip_features.drop(columns=["hour_bin_first"], inplace=True)

    trip_features["straight_line_distance"] = np.sqrt(
        (trip_features["vehicle_x_last"] - trip_features["vehicle_x_first"]) ** 2
        + (trip_features["vehicle_y_last"] - trip_features["vehicle_y_first"]) ** 2
    )

    trip_features["spatial_extent"] = np.sqrt(
        (trip_features["vehicle_x_max"] - trip_features["vehicle_x_min"]) ** 2
        + (trip_features["vehicle_y_max"] - trip_features["vehicle_y_min"]) ** 2
    )
    trip_features.drop(columns=["vehicle_x_max", "vehicle_x_min", "vehicle_y_max", "vehicle_y_min"], inplace=True)

    trip_features.reset_index(inplace=True)
    return trip_features


def fit_source_destination_kmeans(
    train_features: pd.DataFrame,
    n_start_clusters: int = N_START_CLUSTERS,
    n_end_clusters: int = N_END_CLUSTERS,
    random_state: int = RANDOM_SEED_DEFAULT,
) -> dict:
    """
    Fit KMeans clustering models for trip source and destination points.

    Args:
        train_features (pd.DataFrame): Training trip features DataFrame.
        n_start_clusters (int): Number of clusters for starting points.
        n_end_clusters (int): Number of clusters for destination points.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: Dictionary containing fitted KMeans models.
    """
    start_points = train_features[["vehicle_x_first", "vehicle_y_first"]].values
    end_points = train_features[["vehicle_x_last", "vehicle_y_last"]].values

    start_kmeans = KMeans(n_clusters=n_start_clusters, random_state=random_state, n_init=10)
    start_kmeans.fit(start_points)

    end_kmeans = KMeans(n_clusters=n_end_clusters, random_state=random_state, n_init=10)
    end_kmeans.fit(end_points)

    return {
        "start_kmeans": start_kmeans,
        "end_kmeans": end_kmeans,
        "n_start_clusters": n_start_clusters,
        "n_end_clusters": n_end_clusters,
    }


def add_start_end_clusters(trip_features: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Add one-hot encoded cluster features using fitted KMeans models.

    Args:
        trip_features (pd.DataFrame): Trip features DataFrame.
        models (dict): Dictionary containing fitted KMeans models.

    Returns:
        pd.DataFrame: DataFrame with added one-hot encoded cluster columns.
    """
    features_with_clusters = trip_features.copy()

    start_points = features_with_clusters[["vehicle_x_first", "vehicle_y_first"]].values
    end_points = features_with_clusters[["vehicle_x_last", "vehicle_y_last"]].values

    start_clusters = models["start_kmeans"].predict(start_points)
    end_clusters = models["end_kmeans"].predict(end_points)

    n_start = models["n_start_clusters"]
    n_end = models["n_end_clusters"]

    start_onehot = np.zeros((len(start_clusters), n_start), dtype=int)
    start_onehot[np.arange(len(start_clusters)), start_clusters] = 1
    start_cluster_cols = [f"start_cluster_{i}" for i in range(n_start)]
    start_encoded_df = pd.DataFrame(start_onehot, columns=start_cluster_cols, index=features_with_clusters.index)

    end_onehot = np.zeros((len(end_clusters), n_end), dtype=int)
    end_onehot[np.arange(len(end_clusters)), end_clusters] = 1
    end_cluster_cols = [f"end_cluster_{i}" for i in range(n_end)]
    end_encoded_df = pd.DataFrame(end_onehot, columns=end_cluster_cols, index=features_with_clusters.index)

    features_with_clusters = pd.concat([features_with_clusters, start_encoded_df, end_encoded_df], axis=1)

    return features_with_clusters


@dataclass(frozen=True, slots=True)
class FeatureCalibratorFuel:
    """
    Feature calibrator to fit on train trips and transform test and rain trips for fuel consumption prediction.

    Attributes:
        clustering_models (dict): Dictionary containing fitted KMeans models for spatial clustering.
        feature_columns (list[str]): List of feature column names used for modeling.
        n_start_clusters (int): Number of start location clusters.
        n_end_clusters (int): Number of end location clusters.
    """

    clustering_models: dict
    feature_columns: list[str]
    n_start_clusters: int
    n_end_clusters: int

    @classmethod
    def from_train_fcd(
        cls,
        df: pd.DataFrame,
        n_start_clusters: int = N_START_CLUSTERS,
        n_end_clusters: int = N_END_CLUSTERS,
        random_seed: int = RANDOM_SEED_DEFAULT,
    ) -> "FeatureCalibratorFuel":
        """
        Create and fit a FeatureCalibratorFuel for fuel consumption prediction from training trips.

        Args:
            df (pd.DataFrame): DataFrame with training trips (FCD format).
            n_start_clusters (int): Number of clusters for starting points.
            n_end_clusters (int): Number of clusters for destination points.
            random_seed (int): Random seed for reproducibility.

        Returns:
            FeatureCalibratorFuel: New fitted calibrator.

        Raises:
            ValueError: If required columns are not found in the DataFrame.
        """
        required_columns = [
            "vehicle_id",
            "timestep_time",
            "vehicle_x",
            "vehicle_y",
            "vehicle_odometer",
            "vehicle_fuel",
            "hour_bin",
        ]

        if "hour_bin" not in df.columns:
            df = add_time_features(df)

        check_required_columns(df, required_columns, "FCD processing")

        df_with_trips = create_trips_simple(df, min_trip_points=MIN_TRIP_POINTS)
        trip_features = create_trip_features(df_with_trips)
        trip_features = trip_features[trip_features["start_hour"] < START_HOUR_MAX]

        clustering_models = fit_source_destination_kmeans(trip_features, n_start_clusters, n_end_clusters, random_seed)
        trip_features_with_clusters = add_start_end_clusters(trip_features, clustering_models)

        core_feature_cols = [c for c in CORE_FEATURES if c in trip_features_with_clusters.columns]
        start_cluster_cols = [
            c
            for c in trip_features_with_clusters.columns
            if c.startswith("start_cluster_") and not c.endswith("_label")
        ]
        end_cluster_cols = [
            c for c in trip_features_with_clusters.columns if c.startswith("end_cluster_") and not c.endswith("_label")
        ]

        feature_columns = core_feature_cols + start_cluster_cols + end_cluster_cols
        feature_columns = list(dict.fromkeys(feature_columns))

        return cls(
            clustering_models=clustering_models,
            feature_columns=feature_columns,
            n_start_clusters=n_start_clusters,
            n_end_clusters=n_end_clusters,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated feature pipeline to trips.

        Args:
            df (pd.DataFrame): DataFrame with FCD data or trip data.

        Returns:
            pd.DataFrame: DataFrame with engineered features ready for modeling.

        Raises:
            ValueError: If required columns are not found in the DataFrame.
        """
        if "trip_actual_distance" not in df.columns:
            if "hour_bin" not in df.columns:
                df = add_time_features(df)

            required_columns = [
                "vehicle_id",
                "timestep_time",
                "vehicle_x",
                "vehicle_y",
                "vehicle_odometer",
                "vehicle_fuel",
                "hour_bin",
            ]
            check_required_columns(df, required_columns, "FCD processing in transform")

            df_with_trips = create_trips_simple(df, min_trip_points=MIN_TRIP_POINTS)
            trip_features = create_trip_features(df_with_trips)
        else:
            trip_features = df.copy()

        if "start_hour" not in trip_features.columns:
            trip_features["start_hour"] = (trip_features["timestep_time_min"] // 3600) % 24

        if "straight_line_distance" not in trip_features.columns:
            trip_features["straight_line_distance"] = np.sqrt(
                (trip_features["vehicle_x_last"] - trip_features["vehicle_x_first"]) ** 2
                + (trip_features["vehicle_y_last"] - trip_features["vehicle_y_first"]) ** 2
            )

        if "spatial_extent" not in trip_features.columns:
            trip_features["spatial_extent"] = np.sqrt(
                (trip_features["maximum_x"] - trip_features["minimum_x"]) ** 2
                + (trip_features["maximum_y"] - trip_features["minimum_y"]) ** 2
            )

        trip_features.drop(columns=["minimum_x", "maximum_x", "minimum_y", "maximum_y"], inplace=True, errors="ignore")

        trip_features = trip_features[trip_features["start_hour"] < START_HOUR_MAX]
        trip_features_with_clusters = add_start_end_clusters(trip_features, self.clustering_models)

        available_features = [c for c in self.feature_columns if c in trip_features_with_clusters.columns]
        cols = list(available_features)

        for tgt in ["vehicle_fuel_log_sum"]:
            if tgt in trip_features_with_clusters.columns:
                cols.append(tgt)

        return trip_features_with_clusters[cols]

    def save(self, misc_dir: Path) -> None:
        """
        Save the FeatureCalibratorFuel to the misc directory.

        Args:
            misc_dir (Path): Directory to save the FeatureCalibratorFuel to.
        """
        misc_dir.mkdir(parents=True, exist_ok=True)
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        joblib.dump(self, calibrator_path)

        logger.info(f"FeatureCalibratorFuel saved to {calibrator_path}")

    @classmethod
    def load(cls, misc_dir: Path) -> "FeatureCalibratorFuel":
        """
        Load the FeatureCalibratorFuel from the misc directory.

        Args:
            misc_dir (Path): Directory to load the FeatureCalibratorFuel from.

        Returns:
            FeatureCalibratorFuel: Loaded FeatureCalibratorFuel.

        Raises:
            FileNotFoundError: If the FeatureCalibratorFuel file is not found.
            TypeError: If the loaded object is not a FeatureCalibratorFuel.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        if not calibrator_path.exists():
            error_msg = f"FeatureCalibratorFuel file not found: {calibrator_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        feature_calibrator = joblib.load(calibrator_path)
        if not isinstance(feature_calibrator, cls):
            error_msg = f"Loaded object is not a {cls.__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"FeatureCalibratorFuel loaded from {calibrator_path}")

        return feature_calibrator
