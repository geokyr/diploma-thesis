"""Feature engineering and transformation utilities for ETA prediction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from thesis.common.config import (
    AFTERNOON_FLOOR,
    APPDATA_DIRNAME,
    CELL,
    COORDINATE_SCALE,
    FEATURE_CALIBRATOR_FILENAME,
    FEATURE_SELECTION_RESULTS_FILENAME,
    MISC_DIRNAME,
    MORNING_CEILING,
    N_CLUSTERS,
    N_COMPONENTS,
    NOON_CEILING,
    NOON_FLOOR,
    NUM_FREQS,
    PERCENTILE_THRESHOLDS,
    PROJECT_DIR,
    RANDOM_SEED_DEFAULT,
    RUSH_HOURS,
    TARGET_COLUMN,
)
from thesis.common.enums import FeatureGroup, MLTask
from thesis.eta.models import ModelType

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


def _log_feature_addition(df_original: pd.DataFrame, df_final: pd.DataFrame, feature_type: str) -> None:
    """
    Log the number of features added.

    Args:
        df_original (pd.DataFrame): Original DataFrame before adding features.
        df_final (pd.DataFrame): Final DataFrame after adding features.
        feature_type (str): Type of features that were added.
    """
    n_initial_features = len(df_original.columns)
    n_final_features = len(df_final.columns)
    n_added_features = n_final_features - n_initial_features

    logger.info(f"Added {n_added_features} {feature_type} features, for a total of {n_final_features} features")


def split_features_and_target(
    df: pd.DataFrame, target_columns: list[str] = [TARGET_COLUMN]
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """
    Split the features and the target.

    Args:
        df (pd.DataFrame): A DataFrame containing the prepared data.
        target_columns (list[str]): List of column names to use as target.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]: A tuple containing the features and the target.

    Raises:
        ValueError: If the DataFrame does not contain the target columns.
    """
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"DataFrame must contain target columns: {missing_columns}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    X = df.drop(target_columns, axis=1)
    y = df[target_columns]

    n_targets = len(target_columns)
    if n_targets == 1:
        y = y.iloc[:, 0]

    n_samples = len(df)
    variable_word = "variable" if n_targets == 1 else "variables"
    logger.info(f"Split {n_samples} samples into {len(X.columns)} features and {n_targets} target {variable_word}")
    return X, y


def log_transform_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, feature_names: list[str] = ["distance"]
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Log transform features.

    Args:
        X_train (pd.DataFrame): Training features to log transform.
        X_test (pd.DataFrame | None): Test features to log transform.
        feature_names (list[str]): List of feature names to log transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Log transformed training and optionally test features.
    """
    X_train_log = X_train.copy()
    X_train_log[feature_names] = np.log1p(X_train[feature_names])

    if X_test is not None:
        X_test_log = X_test.copy()
        X_test_log[feature_names] = np.log1p(X_test[feature_names])
        logger.info(f"Log transformed {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train_log, X_test_log
    else:
        logger.info(f"Log transformed {len(X_train)} training samples")
        return X_train_log, None


def standard_scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Standard scale training and optionally test features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features to fit scaler on.
        X_test (pd.DataFrame | None): Test features to transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | None]: Scaled training features and optionally scaled test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        logger.info(f"Standard scaled {len(X_train)} training samples and {len(X_test)} test samples")
        return X_train_scaled, X_test_scaled
    else:
        logger.info(f"Standard scaled {len(X_train)} training samples")
        return X_train_scaled, None


def create_log_transformer() -> FunctionTransformer:
    """
    Create a log transformer with exponential inverse function.

    Returns:
        FunctionTransformer: Log transformer with exponential inverse function.
    """
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=True)


def create_quantile_transformer(random_seed: int = RANDOM_SEED_DEFAULT) -> QuantileTransformer:
    """
    Create a quantile transformer.

    Args:
        random_seed (int): The random seed to use for the random number generator.

    Returns:
        QuantileTransformer: Quantile transformer.
    """
    return QuantileTransformer(output_distribution="normal", random_state=random_seed)


def create_boxcox_transformer() -> PowerTransformer:
    """
    Create a box cox transformer.

    Returns:
        PowerTransformer: Box cox transformer.
    """
    return PowerTransformer(method="box-cox")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing time_start column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["time_start"]
    check_required_columns(df, required_columns, "temporal")

    df_hour = df.copy()

    df_hour["hour_bin"] = (df_hour["time_start"] % 36000) // 3600
    df_hour["is_morning"] = (df_hour["hour_bin"] <= MORNING_CEILING).astype(int)
    df_hour["is_noon"] = ((df_hour["hour_bin"] >= NOON_FLOOR) & (df_hour["hour_bin"] <= NOON_CEILING)).astype(int)
    df_hour["is_afternoon"] = (df_hour["hour_bin"] >= AFTERNOON_FLOOR).astype(int)
    df_hour["is_rush_hour"] = df_hour["hour_bin"].isin(RUSH_HOURS).astype(int)

    _log_feature_addition(df, df_hour, "temporal")

    return df_hour


def add_spatial_features(
    df: pd.DataFrame,
    distance_percentiles: np.ndarray | None = None,
    percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
    city_center_x: float | None = None,
    city_center_y: float | None = None,
) -> pd.DataFrame:
    """
    Add spatial features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        distance_percentiles (np.ndarray | None): Percentiles to use for distance features.
        percentile_thresholds (list[int]): Percentiles to use for distance features.
        city_center_x (float | None): Mean center x to use for radial features.
        city_center_y (float | None): Mean center y to use for radial features.

    Returns:
        pd.DataFrame: DataFrame with added spatial features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
    check_required_columns(df, required_columns, "spatial")

    df_spatial = df.copy()

    df_spatial["x_center"] = (df_spatial["source_x"] + df_spatial["destination_x"]) / 2
    df_spatial["y_center"] = (df_spatial["source_y"] + df_spatial["destination_y"]) / 2
    df_spatial["x_difference"] = df_spatial["destination_x"] - df_spatial["source_x"]
    df_spatial["y_difference"] = df_spatial["destination_y"] - df_spatial["source_y"]

    df_spatial["euclidean_distance"] = np.hypot(df_spatial["x_difference"], df_spatial["y_difference"])
    df_spatial["route_efficiency"] = df_spatial["euclidean_distance"] / df_spatial["distance"]
    df_spatial["detour_length"] = df_spatial["distance"] - df_spatial["euclidean_distance"]

    df_spatial["trip_bearing"] = np.arctan2(df_spatial["y_difference"], df_spatial["x_difference"])
    df_spatial["trip_bearing_sin"] = np.sin(df_spatial["trip_bearing"])
    df_spatial["trip_bearing_cos"] = np.cos(df_spatial["trip_bearing"])

    if distance_percentiles is None:
        distance_percentiles = np.percentile(df_spatial["distance"], percentile_thresholds)

    df_spatial["is_short_distance"] = (df_spatial["distance"] <= distance_percentiles[0]).astype(int)
    df_spatial["is_medium_distance"] = (
        (df_spatial["distance"] > distance_percentiles[0]) & (df_spatial["distance"] <= distance_percentiles[2])
    ).astype(int)
    df_spatial["is_long_distance"] = (df_spatial["distance"] > distance_percentiles[2]).astype(int)

    if city_center_x is None:
        city_center_x = np.mean(df_spatial["x_center"])
    if city_center_y is None:
        city_center_y = np.mean(df_spatial["y_center"])

    df_spatial["source_distance_from_city_center"] = np.hypot(
        df_spatial["source_x"] - city_center_x, df_spatial["source_y"] - city_center_y
    )
    df_spatial["destination_distance_from_city_center"] = np.hypot(
        df_spatial["destination_x"] - city_center_x, df_spatial["destination_y"] - city_center_y
    )
    df_spatial["trip_centrality_change"] = (
        df_spatial["destination_distance_from_city_center"] - df_spatial["source_distance_from_city_center"]
    )
    df_spatial["trip_centrality"] = np.hypot(
        df_spatial["x_center"] - city_center_x, df_spatial["y_center"] - city_center_y
    )

    _log_feature_addition(df, df_spatial, "spatial")

    return df_spatial


def add_fourier_features(
    df: pd.DataFrame, coordinate_scale: int = COORDINATE_SCALE, num_freqs: int = NUM_FREQS
) -> pd.DataFrame:
    """
    Add fourier positional encoding features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        coordinate_scale (int): Scale factor to normalize coordinates for fourier encoding.
        num_freqs (int): Number of frequency components to use for fourier encoding.

    Returns:
        pd.DataFrame: DataFrame with added fourier features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "fourier")

    df_fourier = df.copy()

    coordinate_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    for column in coordinate_columns:
        normalized_coordinates = df_fourier[column] / coordinate_scale

        freqs = np.array([2**i for i in range(num_freqs)])

        for i, freq in enumerate(freqs):
            df_fourier[f"{column}_sin_{i}"] = np.sin(freq * normalized_coordinates)
            df_fourier[f"{column}_cos_{i}"] = np.cos(freq * normalized_coordinates)

    _log_feature_addition(df, df_fourier, "fourier")

    return df_fourier


def add_cell_features(df: pd.DataFrame, cell: int = CELL) -> pd.DataFrame:
    """
    Add cell features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        cell (int): Size of the cell in meters.

    Returns:
        pd.DataFrame: DataFrame with added cell features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "cell")

    df_cell = df.copy()

    df_cell["source_cell_x"] = (df_cell["source_x"] // cell).astype(int)
    df_cell["source_cell_y"] = (df_cell["source_y"] // cell).astype(int)
    df_cell["destination_cell_x"] = (df_cell["destination_x"] // cell).astype(int)
    df_cell["destination_cell_y"] = (df_cell["destination_y"] // cell).astype(int)

    _log_feature_addition(df, df_cell, "cell")

    return df_cell


def add_cluster_features(
    df: pd.DataFrame,
    kmeans_source: KMeans | None = None,
    kmeans_destination: KMeans | None = None,
    n_clusters: int = N_CLUSTERS,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add cluster features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        kmeans_source (KMeans | None): KMeans object to use for source cluster.
        kmeans_destination (KMeans | None): KMeans object to use for destination cluster.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering.

    Returns:
        pd.DataFrame: DataFrame with added cluster features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "cluster")

    df_cluster = df.copy()

    source_coordinates = df_cluster[["source_x", "source_y"]].to_numpy()
    destination_coordinates = df_cluster[["destination_x", "destination_y"]].to_numpy()
    if kmeans_source is None:
        kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
        df_cluster["source_cluster"] = kmeans_source.fit_predict(source_coordinates)
    else:
        df_cluster["source_cluster"] = kmeans_source.predict(source_coordinates)

    if kmeans_destination is None:
        kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)
        df_cluster["destination_cluster"] = kmeans_destination.fit_predict(destination_coordinates)
    else:
        df_cluster["destination_cluster"] = kmeans_destination.predict(destination_coordinates)

    _log_feature_addition(df, df_cluster, "cluster")

    return df_cluster


def add_pca_features(
    df: pd.DataFrame,
    pca_coordinates: PCA | None = None,
    n_components: int = N_COMPONENTS,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add pca features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        pca_coordinates (PCA | None): PCA object to use for pca.
        n_components (int): Number of components for pca.
        random_seed (int): Random seed for pca.

    Returns:
        pd.DataFrame: DataFrame with added pca features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    check_required_columns(df, required_columns, "pca")

    df_pca = df.copy()

    source_coordinates = df_pca[["source_x", "source_y"]].to_numpy()
    destination_coordinates = df_pca[["destination_x", "destination_y"]].to_numpy()

    if pca_coordinates is None:
        all_coordinates = np.vstack([source_coordinates, destination_coordinates])
        pca_coordinates = PCA(n_components=n_components, random_state=random_seed)
        pca_coordinates.fit(all_coordinates)

    source_coordinates_pca = pca_coordinates.transform(source_coordinates)
    destination_coordinates_pca = pca_coordinates.transform(destination_coordinates)

    df_pca["source_pca_1"] = source_coordinates_pca[:, 0]
    df_pca["source_pca_2"] = source_coordinates_pca[:, 1]
    df_pca["dest_pca_1"] = destination_coordinates_pca[:, 0]
    df_pca["dest_pca_2"] = destination_coordinates_pca[:, 1]

    _log_feature_addition(df, df_pca, "pca")

    return df_pca


# TODO: remove this wrapper
def add_all_features(
    df: pd.DataFrame,
    distance_percentiles: np.ndarray | None = None,
    percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
    city_center_x: float | None = None,
    city_center_y: float | None = None,
    coordinate_scale: int = COORDINATE_SCALE,
    num_freqs: int = NUM_FREQS,
    cell: int = CELL,
    kmeans_source: KMeans | None = None,
    kmeans_destination: KMeans | None = None,
    n_clusters: int = N_CLUSTERS,
    random_seed: int = RANDOM_SEED_DEFAULT,
    pca_coordinates: PCA | None = None,
    n_components: int = N_COMPONENTS,
) -> pd.DataFrame:
    """
    Add all features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        distance_percentiles (np.ndarray | None): Percentiles to use for distance features.
        percentile_thresholds (list[int]): Percentiles to use for distance features.
        city_center_x (float | None): Mean center x to use for radial features.
        city_center_y (float | None): Mean center y to use for radial features.
        coordinate_scale (int): Scale factor to normalize coordinates for fourier encoding.
        num_freqs (int): Number of frequency components to use for fourier encoding.
        cell (int): Size of the cell in meters.
        kmeans_source (KMeans | None): KMeans object to use for source cluster.
        kmeans_destination (KMeans | None): KMeans object to use for destination cluster.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering and pca.
        pca_coordinates (PCA | None): PCA object to use for pca.
        n_components (int): Number of components for pca.

    Returns:
        pd.DataFrame: DataFrame with added all features.

    Raises:
        ValueError: If the required columns are not found in the DataFrame.
    """
    df = add_temporal_features(df)
    df = add_spatial_features(df, distance_percentiles, percentile_thresholds, city_center_x, city_center_y)
    df = add_fourier_features(df, coordinate_scale, num_freqs)
    df = add_cell_features(df, cell)
    df = add_cluster_features(df, kmeans_source, kmeans_destination, n_clusters, random_seed)
    df = add_pca_features(df, pca_coordinates, n_components, random_seed)

    return df


@dataclass(frozen=True, slots=True)
class FeatureCalibrator:
    """
    Feature calibrator to fit on train trips and transform test and rain trips.

    Attributes:
        feature_groups (list[FeatureGroup]): List of feature groups to apply
        distance_percentiles (np.ndarray | None): Thresholds at configured percentiles
        city_center_x (float | None): Mean center x used for radial features
        city_center_y (float | None): Mean center y used for radial features
        kmeans_source (KMeans | None): Clustering on source coordinates
        kmeans_destination (KMeans | None): Clustering on destination coordinates
        pca_coordinates (PCA | None): PCA on coordinates
    """

    feature_groups: tuple[FeatureGroup, ...]
    distance_percentiles: np.ndarray | None = None
    city_center_x: float | None = None
    city_center_y: float | None = None
    kmeans_source: KMeans | None = None
    kmeans_destination: KMeans | None = None
    pca_coordinates: PCA | None = None

    @classmethod
    def from_train_trips(
        cls,
        df: pd.DataFrame,
        feature_groups: tuple[FeatureGroup, ...],
        percentile_thresholds: list[int] = PERCENTILE_THRESHOLDS,
        n_clusters: int = N_CLUSTERS,
        random_seed: int = RANDOM_SEED_DEFAULT,
        n_components: int = N_COMPONENTS,
    ) -> "FeatureCalibrator":
        """
        Create and fit a FeatureCalibrator from training trips.

        Args:
            df (pd.DataFrame): DataFrame with training trips.
            feature_groups (list[FeatureGroup]): List of feature groups to fit and transform.
            percentile_thresholds (list[int]): Percentiles to use for distance features.
            n_clusters (int): Number of clusters for K-means clustering on coordinates.
            random_seed (int): Random seed for clustering and pca.
            n_components (int): Number of components for pca.

        Returns:
            FeatureCalibrator: New fitted FeatureCalibrator.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        distance_percentiles = None
        city_center_x = None
        city_center_y = None
        kmeans_source = None
        kmeans_destination = None
        pca_coordinates = None

        if FeatureGroup.SPATIAL in feature_groups:
            required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
            check_required_columns(df, required_columns, "spatial calibration")

            distance_percentiles = np.percentile(df["distance"], percentile_thresholds)

            x_center = (df["source_x"] + df["destination_x"]) / 2
            y_center = (df["source_y"] + df["destination_y"]) / 2
            city_center_x = np.mean(x_center)
            city_center_y = np.mean(y_center)

        if FeatureGroup.CLUSTER in feature_groups:
            required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
            check_required_columns(df, required_columns, "cluster calibration")

            source_coordinates = df[["source_x", "source_y"]].to_numpy()
            destination_coordinates = df[["destination_x", "destination_y"]].to_numpy()
            kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
            kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)
            kmeans_source.fit(source_coordinates)
            kmeans_destination.fit(destination_coordinates)

        if FeatureGroup.PCA in feature_groups:
            required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
            check_required_columns(df, required_columns, "pca calibration")

            source_coordinates = df[["source_x", "source_y"]].to_numpy()
            destination_coordinates = df[["destination_x", "destination_y"]].to_numpy()
            all_coordinates = np.vstack([source_coordinates, destination_coordinates])
            pca_coordinates = PCA(n_components=n_components, random_state=random_seed)
            pca_coordinates.fit(all_coordinates)

        return cls(
            feature_groups=feature_groups,
            distance_percentiles=distance_percentiles,
            city_center_x=city_center_x,
            city_center_y=city_center_y,
            kmeans_source=kmeans_source,
            kmeans_destination=kmeans_destination,
            pca_coordinates=pca_coordinates,
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibrated feature pipeline to trips.

        Args:
            df (pd.DataFrame): DataFrame with trip data.

        Returns:
            pd.DataFrame: DataFrame with added calibrated features.

        Raises:
            ValueError: If the required columns are not found in the DataFrame.
        """
        if FeatureGroup.TEMPORAL in self.feature_groups:
            df = add_temporal_features(df)

        if FeatureGroup.SPATIAL in self.feature_groups:
            df = add_spatial_features(
                df,
                distance_percentiles=self.distance_percentiles,
                city_center_x=self.city_center_x,
                city_center_y=self.city_center_y,
            )

        if FeatureGroup.FOURIER in self.feature_groups:
            df = add_fourier_features(df)

        if FeatureGroup.CELL in self.feature_groups:
            df = add_cell_features(df)

        if FeatureGroup.CLUSTER in self.feature_groups:
            df = add_cluster_features(
                df,
                kmeans_source=self.kmeans_source,
                kmeans_destination=self.kmeans_destination,
            )

        if FeatureGroup.PCA in self.feature_groups:
            df = add_pca_features(df, pca_coordinates=self.pca_coordinates)

        return df

    def save(self, misc_dir: Path) -> None:
        """
        Save the FeatureCalibrator to the misc directory.

        Args:
            misc_dir (Path): Directory to save the FeatureCalibrator to.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        joblib.dump(self, calibrator_path)

        logger.info(f"FeatureCalibrator saved to {calibrator_path}")

    @classmethod
    def load(cls, misc_dir: Path) -> "FeatureCalibrator":
        """
        Load the FeatureCalibrator from the misc directory.

        Args:
            misc_dir (Path): Directory to load the FeatureCalibrator from.

        Returns:
            FeatureCalibrator: Loaded FeatureCalibrator.

        Raises:
            FileNotFoundError: If the FeatureCalibrator file does not exist.
            TypeError: If the loaded object is not a FeatureCalibrator.
        """
        calibrator_path = misc_dir / FEATURE_CALIBRATOR_FILENAME
        if not calibrator_path.exists():
            error_msg = f"FeatureCalibrator file not found: {calibrator_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        feature_calibrator = joblib.load(calibrator_path)
        if not isinstance(feature_calibrator, cls):
            error_msg = f"Loaded object is not a {cls.__name__}"
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.info(f"FeatureCalibrator loaded from {calibrator_path}")

        return feature_calibrator

    def get_feature_calibrator_dir(self, ml_task: MLTask) -> Path:
        """
        Get the path to the feature calibrator directory.

        Args:
            ml_task (MLTask): ML task.

        Returns:
            Path: Path to the feature calibrator directory.
        """
        return PROJECT_DIR / APPDATA_DIRNAME / MISC_DIRNAME / ml_task


def get_gain_feature_importance(model: BaseEstimator, feature_names: list[str]) -> pd.DataFrame:
    """
    Get the gain feature importance of the model.

    Args:
        model (BaseEstimator): Model to get the gain feature importance of.
        feature_names (list[str]): List of feature names.
    """
    gain_raw = model.feature_importances_

    gain = np.asarray(gain_raw, dtype=float)
    total_gain = gain.sum()
    percentage_gain = (gain / total_gain * 100.0) if total_gain > 0 else np.zeros_like(gain)

    return (
        pd.DataFrame({"feature": feature_names, "importance": gain, "percentage": percentage_gain})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_average_gain_feature_importance(per_fold_gain_feature_importances: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Get the average gain feature importance across CV folds.

    Args:
        per_fold_gain_feature_importances (list[pd.DataFrame]): List of per fold gain feature importances.

    Returns:
        pd.DataFrame: Average gain feature importance.
    """
    per_fold_gain_feature_importances = pd.concat(per_fold_gain_feature_importances)

    average_gain_feature_importance = (
        per_fold_gain_feature_importances.groupby("feature")
        .agg({"importance": "mean", "percentage": "mean"})
        .reset_index()
    )
    average_gain_feature_importance.columns = ["feature", "importance_mean", "percentage_mean"]
    average_gain_feature_importance = average_gain_feature_importance.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    logger.info(
        f"Ranked features by average gain feature importance across CV folds\n{average_gain_feature_importance.to_string(index=False)}"
    )

    logger.info("Cumulative importance")
    logger.info(f"  Top 10: {average_gain_feature_importance.head(10)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 20: {average_gain_feature_importance.head(20)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 30: {average_gain_feature_importance.head(30)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 40: {average_gain_feature_importance.head(40)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 50: {average_gain_feature_importance.head(50)['percentage_mean'].sum():5.1f}%")

    return average_gain_feature_importance


def combine_model_feature_importances(
    model_importances: dict[ModelType, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine feature importances from multiple models into a single DataFrame.

    Args:
        model_importances (dict[ModelType, pd.DataFrame]): Dictionary of model types and average gain feature importance DataFrames.

    Returns:
        pd.DataFrame: Combined feature importances with a ModelType column.
    """
    combined_results = []

    for model_type, importance_df in model_importances.items():
        df_copy = importance_df.copy()
        df_copy.insert(0, "model", model_type)
        combined_results.append(df_copy)

    combined_df = pd.concat(combined_results, ignore_index=True)

    logger.info(f"Combined feature importances from {len(model_importances)} models")

    return combined_df


def save_feature_selection_results(feature_selection_results: pd.DataFrame, results_dir: Path) -> None:
    """
    Save the feature selection results to the results directory.

    Args:
        feature_selection_results (pd.DataFrame): Feature selection results.
        results_dir (Path): Directory to save the feature selection results to.
    """
    feature_selection_results_path = results_dir / FEATURE_SELECTION_RESULTS_FILENAME
    feature_selection_results.to_csv(feature_selection_results_path, index=False)

    logger.info(f"Feature selection results saved to {feature_selection_results_path}")
