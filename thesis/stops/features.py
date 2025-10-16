"""Feature engineering and transformation utilities for Stops prediction."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from thesis.common.config import (
    FEATURE_CALIBRATOR_FILENAME,
    MIN_DISTANCE_STOPS,
    MIN_DURATION_STOPS,
    MIN_TRIP_RECORDS_STOPS,
    N_CLUSTERS_STOPS,
    N_INIT_STOPS,
    RANDOM_SEED_STOPS,
    TARGET_COLUMN_STOPS,
    USE_ROUTE_FEATURES_STOPS,
    USE_SPATIAL_FEATURES_STOPS,
)

logger = logging.getLogger(__name__)


def check_required_columns(df: pd.DataFrame, required_columns: list[str], feature_type: str) -> None:
    """
    Check if DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): DataFrame to check.
        required_columns (list[str]): List of required column names.
        feature_type (str): Type of features being added for logging.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        error_msg = f"Missing columns {missing_columns} while adding {feature_type} features"
        logger.error(error_msg)
        raise ValueError(error_msg)


def extract_trip_features(
    df: pd.DataFrame,
    min_trip_records: int,
    min_duration: float,
    min_distance: float,
    extract_routes: bool,
) -> dict[str, str | int | float] | None:
    """
    Extract trip-level features from a single trip's FCD records.

    Args:
        df (pd.DataFrame): FCD records for a single trip.
        min_trip_records (int): Minimum number of records required.
        min_duration (float): Minimum trip duration in seconds.
        min_distance (float): Minimum trip distance in meters.
        extract_routes (bool): Whether to extract route features.

    Returns:
        dict[str, str | int | float] | None: Dictionary with trip features or None if trip is invalid.
    """
    if len(df) < min_trip_records:
        return None

    df = df.sort_values("timestep_time")
    start_record, end_record = df.iloc[0], df.iloc[-1]

    trip_start_x, trip_start_y = start_record["vehicle_x"], start_record["vehicle_y"]
    trip_end_x, trip_end_y = end_record["vehicle_x"], end_record["vehicle_y"]
    actual_distance = end_record["vehicle_odometer"] - start_record["vehicle_odometer"]
    straight_distance = np.hypot(trip_end_x - trip_start_x, trip_end_y - trip_start_y)
    duration = end_record["timestep_time"] - start_record["timestep_time"]

    if duration <= min_duration or actual_distance <= min_distance:
        return None

    stopped_series = df["vehicle_speed"] == 0
    if stopped_series.sum() == 0:
        actual_stops = 0
    else:
        stop_groups = (stopped_series != stopped_series.shift()).cumsum()
        actual_stops = stop_groups[stopped_series].nunique()

    hour_of_day = int((start_record["timestep_time"] // 3600) % 24)

    route_num_edges = 0
    route_unique_edges = 0
    route_edge_reuse_ratio = 1.0
    if extract_routes and "edge_id" in df.columns:
        edges = df["edge_id"].dropna()
        if len(edges) > 0:
            route_num_edges = len(edges)
            route_unique_edges = edges.nunique()
            route_edge_reuse_ratio = route_unique_edges / route_num_edges

    return {
        "trip_id": start_record["trip_id"],
        "vehicle_id": start_record["vehicle_id"],
        "trip_start_x": trip_start_x,
        "trip_start_y": trip_start_y,
        "trip_end_x": trip_end_x,
        "trip_end_y": trip_end_y,
        "actual_distance": actual_distance,
        "straight_distance": straight_distance,
        "actual_stops": actual_stops,
        "hour_of_day": hour_of_day,
        "timestep": start_record["timestep_time"],
        "route_num_edges": route_num_edges,
        "route_unique_edges": route_unique_edges,
        "route_edge_reuse_ratio": route_edge_reuse_ratio,
    }


def fit_spatial_clusters(
    df: pd.DataFrame,
    n_clusters: int,
    random_seed: int,
    target_column: str,
) -> tuple[float, float, KMeans, KMeans, dict[int, float], dict[int, float]]:
    """
    Fit spatial clustering models for source and destination coordinates.

    Args:
        df (pd.DataFrame): DataFrame with trip data.
        n_clusters (int): Number of clusters for spatial clustering.
        random_seed (int): Random seed for reproducibility.
        target_column (str): Name of the target column for target encoding.

    Returns:
        tuple[float, float, KMeans, KMeans, dict[int, float], dict[int, float]]: Tuple containing city center x and y coordinates, fitted KMeans models for source and destination coordinates, target encoding means for source and destination clusters.
    """
    city_center_x = df["trip_start_x"].mean()
    city_center_y = df["trip_start_y"].mean()

    kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=N_INIT_STOPS)
    kmeans_dest = KMeans(n_clusters=n_clusters, random_state=random_seed + 1, n_init=N_INIT_STOPS)

    source_coordinates = df[["trip_start_x", "trip_start_y"]]
    dest_coordinates = df[["trip_end_x", "trip_end_y"]]

    kmeans_source.fit(source_coordinates)
    kmeans_dest.fit(dest_coordinates)

    source_cluster_target_means = df.groupby(kmeans_source.predict(source_coordinates))[target_column].mean().to_dict()
    dest_cluster_target_means = df.groupby(kmeans_dest.predict(dest_coordinates))[target_column].mean().to_dict()

    return (
        city_center_x,
        city_center_y,
        kmeans_source,
        kmeans_dest,
        source_cluster_target_means,
        dest_cluster_target_means,
    )


def add_spatial_features(
    df: pd.DataFrame,
    city_center_x: float,
    city_center_y: float,
    kmeans_source: KMeans,
    kmeans_dest: KMeans,
    source_cluster_target_means: dict[int, float],
    dest_cluster_target_means: dict[int, float],
    n_clusters: int,
) -> pd.DataFrame:
    """
    Add spatial clustering features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with trip data.
        city_center_x (float): Mean x coordinate of city center.
        city_center_y (float): Mean y coordinate of city center.
        kmeans_source (KMeans): KMeans clustering model for source coordinates.
        kmeans_dest (KMeans): KMeans clustering model for destination coordinates.
        source_cluster_target_means (dict[int, float]): Target encoding means for source clusters.
        dest_cluster_target_means (dict[int, float]): Target encoding means for destination clusters.
        n_clusters (int): Number of clusters for spatial clustering.

    Returns:
        pd.DataFrame: DataFrame with added spatial features.
    """
    df["source_dist_to_center"] = np.hypot(df["trip_start_x"] - city_center_x, df["trip_start_y"] - city_center_y)

    source_coords = df[["trip_start_x", "trip_start_y"]]
    df["source_cluster"] = kmeans_source.predict(source_coords)
    df["source_cluster_target_encoded"] = df["source_cluster"].map(source_cluster_target_means).fillna(0)

    source_dummies = pd.get_dummies(df["source_cluster"], prefix="src_cluster").reindex(
        columns=[f"src_cluster_{i}" for i in range(n_clusters)], fill_value=0
    )
    df = pd.concat([df, source_dummies], axis=1)

    for i, center in enumerate(kmeans_source.cluster_centers_):
        df[f"src_dist_to_cluster_{i}"] = np.hypot(df["trip_start_x"] - center[0], df["trip_start_y"] - center[1])

    df["dest_dist_to_center"] = np.hypot(df["trip_end_x"] - city_center_x, df["trip_end_y"] - city_center_y)

    dest_coords = df[["trip_end_x", "trip_end_y"]]
    df["dest_cluster"] = kmeans_dest.predict(dest_coords)
    df["dest_cluster_target_encoded"] = df["dest_cluster"].map(dest_cluster_target_means).fillna(0)

    dest_dummies = pd.get_dummies(df["dest_cluster"], prefix="dst_cluster").reindex(
        columns=[f"dst_cluster_{i}" for i in range(n_clusters)], fill_value=0
    )
    df = pd.concat([df, dest_dummies], axis=1)

    for i, center in enumerate(kmeans_dest.cluster_centers_):
        df[f"dst_dist_to_cluster_{i}"] = np.hypot(df["trip_end_x"] - center[0], df["trip_end_y"] - center[1])

    return df


@dataclass(frozen=True, slots=True)
class FeatureCalibratorStops:
    """
    Feature calibrator to fit on train trips and transform test and rain trips for stops prediction.

    Attributes:
        kmeans_source (KMeans): K-means clustering model for source coordinates.
        kmeans_dest (KMeans): K-means clustering model for destination coordinates.
        city_center_x (float): Mean x coordinate of city center.
        city_center_y (float): Mean y coordinate of city center.
        source_cluster_target_means (dict[int, float]): Target encoding means for source clusters.
        dest_cluster_target_means (dict[int, float]): Target encoding means for destination clusters.
        use_route_features (bool): Whether to use route features.
        use_spatial_features (bool): Whether to use spatial features.
    """

    kmeans_source: KMeans
    kmeans_dest: KMeans
    city_center_x: float
    city_center_y: float
    source_cluster_target_means: dict[int, float]
    dest_cluster_target_means: dict[int, float]
    use_route_features: bool
    use_spatial_features: bool
    n_clusters: int

    _REQUIRED_COLUMNS: ClassVar[list[str]] = [
        "trip_start_x",
        "trip_start_y",
        "trip_end_x",
        "trip_end_y",
        "actual_distance",
        "route_edges",
    ]

    _REQUIRED_FCD_COLUMNS: ClassVar[list[str]] = [
        "vehicle_id",
        "timestep_time",
        "vehicle_x",
        "vehicle_y",
        "vehicle_speed",
        "vehicle_odometer",
    ]

    @classmethod
    def from_train_fcd(
        cls,
        df: pd.DataFrame,
        target_column: str = TARGET_COLUMN_STOPS,
        random_seed: int = RANDOM_SEED_STOPS,
        n_clusters: int = N_CLUSTERS_STOPS,
        min_trip_records: int = MIN_TRIP_RECORDS_STOPS,
        min_duration: float = MIN_DURATION_STOPS,
        min_distance: float = MIN_DISTANCE_STOPS,
        use_spatial_features: bool = USE_SPATIAL_FEATURES_STOPS,
        use_route_features: bool = USE_ROUTE_FEATURES_STOPS,
    ) -> "FeatureCalibratorStops":
        """
        Create and fit a FeatureCalibratorStops from training FCD data.

        Args:
            df (pd.DataFrame): DataFrame with training trips containing required columns.
            target_column (str): Name of the target column for target encoding.
            n_clusters (int): Number of clusters for spatial clustering.
            random_seed (int): Random seed for reproducibility.
            min_trip_records (int): Minimum number of FCD records required per trip.
            min_duration (float): Minimum trip duration in seconds.
            min_distance (float): Minimum trip distance in meters.
            use_spatial_features (bool): Whether to use spatial features.
            use_route_features (bool): Whether to use route features.

        Returns:
            FeatureCalibratorStops: New fitted FeatureCalibratorStops.

        Raises:
            ValueError: If required columns are not found in the DataFrame.
        """

        check_required_columns(df, FeatureCalibratorStops._REQUIRED_FCD_COLUMNS, "FeatureCalibratorStops calibration")

        df = df.sort_values(["vehicle_id", "timestep_time"])
        df["time_diff"] = df.groupby("vehicle_id")["timestep_time"].diff().fillna(np.inf)
        df["trip_segment"] = (df["time_diff"] != 1).groupby(df["vehicle_id"]).cumsum()
        df["trip_id"] = df["vehicle_id"].astype(str) + "_" + df["trip_segment"].astype(str)

        if use_route_features and "vehicle_lane" in df.columns:
            df["edge_id"] = df["vehicle_lane"].astype(str).str.rsplit("_", n=1).str[0]
            df = df[df["edge_id"].notna() & (df["edge_id"] != "")]

        trip_list = []
        for _, grp in df.groupby("trip_id"):
            trip_features = extract_trip_features(grp, min_trip_records, min_duration, min_distance, use_route_features)
            if trip_features:
                trip_list.append(trip_features)

        df_trips = pd.DataFrame(trip_list)

        if use_spatial_features:
            (
                city_center_x,
                city_center_y,
                kmeans_source,
                kmeans_dest,
                source_cluster_target_means,
                dest_cluster_target_means,
            ) = fit_spatial_clusters(df_trips, n_clusters, random_seed, target_column)

        if use_spatial_features:
            df_trips = add_spatial_features(
                df_trips,
                city_center_x,
                city_center_y,
                kmeans_source,
                kmeans_dest,
                source_cluster_target_means,
                dest_cluster_target_means,
                n_clusters,
            )

        logger.info("Calibrated FeatureCalibratorStops")

        return cls(
            kmeans_source=kmeans_source,
            kmeans_dest=kmeans_dest,
            city_center_x=city_center_x,
            city_center_y=city_center_y,
            source_cluster_target_means=source_cluster_target_means,
            dest_cluster_target_means=dest_cluster_target_means,
            use_route_features=use_route_features,
            use_spatial_features=use_spatial_features,
            n_clusters=n_clusters,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated feature pipeline to trips.

        Args:
            df (pd.DataFrame): DataFrame with trip data.

        Returns:
            pd.DataFrame: DataFrame with all features added.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        check_required_columns(df, FeatureCalibratorStops._REQUIRED_COLUMNS, "FeatureCalibratorStops transformation")

        df_final = df.copy()

        if self.use_route_features:
            if "route_num_edges" not in df_final.columns:
                df_final["route_num_edges"] = 0
            if "route_unique_edges" not in df_final.columns:
                df_final["route_unique_edges"] = 0
            if "route_edge_reuse_ratio" not in df_final.columns:
                df_final["route_edge_reuse_ratio"] = 1.0

            for idx, edges in enumerate(df_final["route_edges"]):
                n_edges = len(edges)
                unique_edges = len(set(edges))
                reuse_ratio = unique_edges / n_edges
                df_final.loc[df_final.index[idx], "route_num_edges"] = n_edges
                df_final.loc[df_final.index[idx], "route_unique_edges"] = unique_edges
                df_final.loc[df_final.index[idx], "route_edge_reuse_ratio"] = reuse_ratio

        if self.use_spatial_features:
            if "straight_distance" not in df_final.columns:
                df_final["straight_distance"] = np.hypot(
                    df_final["trip_end_x"] - df_final["trip_start_x"],
                    df_final["trip_end_y"] - df_final["trip_start_y"],
                )

            df_final = add_spatial_features(
                df_final,
                city_center_x=self.city_center_x,
                city_center_y=self.city_center_y,
                kmeans_source=self.kmeans_source,
                kmeans_dest=self.kmeans_dest,
                source_cluster_target_means=self.source_cluster_target_means,
                dest_cluster_target_means=self.dest_cluster_target_means,
                n_clusters=self.n_clusters,
            )

        logger.info(
            f"Transformed {len(df_final)} samples with FeatureCalibratorStops, total features: {len(df_final.columns)}"
        )

        return df_final

    def save(self, misc_dir: Path) -> None:
        """
        Save the FeatureCalibratorStops to the misc directory.

        Args:
            misc_dir (Path): Directory to save the FeatureCalibratorStops to.
        """
        misc_dir.mkdir(parents=True, exist_ok=True)
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        joblib.dump(self, calibrator_path)

        logger.info(f"FeatureCalibratorStops saved to {calibrator_path}")

    @classmethod
    def load(cls, misc_dir: Path) -> "FeatureCalibratorStops":
        """
        Load the FeatureCalibratorStops from the misc directory.

        Args:
            misc_dir (Path): Directory to load the FeatureCalibratorStops from.

        Returns:
            FeatureCalibratorStops: Loaded FeatureCalibratorStops.

        Raises:
            FileNotFoundError: If the FeatureCalibratorStops file is not found.
            TypeError: If the loaded object is not a FeatureCalibratorStops.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        if not calibrator_path.exists():
            error_msg = f"FeatureCalibratorStops file not found: {calibrator_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        feature_calibrator = joblib.load(calibrator_path)
        if not isinstance(feature_calibrator, cls):
            error_msg = f"Loaded object is not a {cls.__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"FeatureCalibratorStops loaded from {calibrator_path}")

        return feature_calibrator
