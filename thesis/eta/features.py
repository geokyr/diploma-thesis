"""Feature engineering and transformation utilities for ETA prediction."""

import logging
from dataclasses import asdict, dataclass
from functools import reduce
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from thesis.common.config import (
    AFTERNOON_FLOOR,
    APPDATA_DIRNAME,
    CELL,
    COORDINATE_SCALE,
    CORRELATION_THRESHOLD,
    FEATURE_CALIBRATOR_FILENAME,
    FEATURE_CATEGORIES,
    FEATURE_SELECTION_RESULTS_FILENAME,
    MISC_DIRNAME,
    MORNING_CEILING,
    N_CLUSTERS,
    N_COMPONENTS,
    NOON_CEILING,
    NOON_FLOOR,
    NUM_FREQS,
    OUTPUTS_DIR,
    PERCENTILE_THRESHOLDS,
    PERMUTATION_IMPORTANCE_N_JOBS,
    PERMUTATION_IMPORTANCE_N_REPEATS,
    PERMUTATION_IMPORTANCE_SCORING,
    PROJECT_DIR,
    RANDOM_SEED_DEFAULT,
    RANKING_ALPHA,
    RESULTS_DIRNAME,
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


def combine_feature_selection_results(
    model_importances: dict[ModelType, pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine feature selection results from multiple models into a single DataFrame.

    Args:
        model_importances (dict[ModelType, pd.DataFrame]): Dictionary of model types and feature selection results DataFrames.

    Returns:
        pd.DataFrame: Combined feature selection results with a ModelType column.
    """
    combined_results = []

    for model_type, importance_df in model_importances.items():
        df_copy = importance_df.copy()
        df_copy.insert(0, "model", model_type)
        combined_results.append(df_copy)

    combined_df = pd.concat(combined_results, ignore_index=True)

    logger.info(f"Combined feature selection results from {len(model_importances)} models")

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


def find_correlated_feature_pairs(X: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD) -> pd.DataFrame:
    """
    Find correlated feature pairs in a DataFrame.

    Args:
        X (pd.DataFrame): DataFrame to find correlated feature pairs in.
        threshold (float): Threshold for correlation.

    Returns:
        pd.DataFrame: DataFrame with correlated feature pairs.
    """
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

    correlated_feature_pairs = []
    for column in upper_triangle.columns:
        correlated = upper_triangle[column][upper_triangle[column] > threshold]
        for index in correlated.index:
            correlated_feature_pairs.append({"feature_1": column, "feature_2": index, "correlation": correlated[index]})

    correlated_feature_pairs = pd.DataFrame(correlated_feature_pairs)

    if len(correlated_feature_pairs) > 0:
        logger.info(
            f"Found {len(correlated_feature_pairs)} correlated feature pairs\n{correlated_feature_pairs.to_string(index=False)}"
        )

        all_correlated_features = set(
            correlated_feature_pairs["feature_1"].tolist() + correlated_feature_pairs["feature_2"].tolist()
        )
        logger.info(f"{len(all_correlated_features)} unique features involved in high correlations")
    else:
        logger.info(f"No correlated feature pairs found with threshold {threshold}")

    return correlated_feature_pairs


def get_permutation_importance(
    model: BaseEstimator,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_repeats: int = PERMUTATION_IMPORTANCE_N_REPEATS,
    scoring: str = PERMUTATION_IMPORTANCE_SCORING,
    n_jobs: int = PERMUTATION_IMPORTANCE_N_JOBS,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Get the permutation importance of the model.

    Args:
        model (BaseEstimator): Model to get the permutation importance of.
        X_validation (pd.DataFrame): Validation features.
        y_validation (pd.Series): Validation target.
        n_repeats (int): Number of repeats.
        n_jobs (int): Number of jobs.
        random_seed (int): Random seed.

    Returns:
        pd.DataFrame: Permutation importance.
    """
    permutation_importances = permutation_importance(
        model, X_validation, y_validation, n_repeats=n_repeats, scoring=scoring, n_jobs=n_jobs, random_state=random_seed
    )

    return pd.DataFrame({"feature": X_validation.columns, "importance": permutation_importances.importances_mean})


def get_average_permutation_importance(per_fold_permutation_importances: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Get the average permutation importance across CV folds.

    Args:
        per_fold_permutation_importances (list[pd.DataFrame]): List of per fold permutation importances.

    Returns:
        pd.DataFrame: Average permutation importance.
    """
    per_fold_permutation_importances = pd.concat(per_fold_permutation_importances)

    average_permutation_importance = (
        per_fold_permutation_importances.groupby("feature").agg({"importance": "mean"}).reset_index()
    )
    average_permutation_importance.columns = ["feature", "importance_mean"]
    average_permutation_importance["percentage_mean"] = (
        average_permutation_importance["importance_mean"]
        / average_permutation_importance["importance_mean"].sum()
        * 100
    )
    average_permutation_importance = average_permutation_importance.sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)

    logger.info(
        f"Ranked features by average permutation importance across CV folds\n{average_permutation_importance.to_string(index=False)}"
    )

    logger.info("Cumulative importance")
    logger.info(f"  Top 10: {average_permutation_importance.head(10)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 20: {average_permutation_importance.head(20)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 30: {average_permutation_importance.head(30)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 40: {average_permutation_importance.head(40)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 50: {average_permutation_importance.head(50)['percentage_mean'].sum():5.1f}%")

    return average_permutation_importance


def get_shap_importance(model: BaseEstimator, X_validation: pd.DataFrame) -> pd.DataFrame:
    """
    Get the SHAP importance of the model.

    Args:
        model (BaseEstimator): Model to get the SHAP importance of.
        X_validation (pd.DataFrame): Validation features.

    Returns:
        pd.DataFrame: SHAP importance with feature names and importance values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_validation)
    shap_importances = np.abs(shap_values).mean(axis=0)

    total_importance = shap_importances.sum()
    percentage = (
        (shap_importances / total_importance * 100.0) if total_importance > 0 else np.zeros_like(shap_importances)
    )

    return (
        pd.DataFrame(
            {
                "feature": X_validation.columns,
                "importance": shap_importances,
                "percentage": percentage,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def get_average_shap_importance(per_fold_shap_importances: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Get the average SHAP importance across CV folds.

    Args:
        per_fold_shap_importances (list[pd.DataFrame]): List of per fold SHAP importances.

    Returns:
        pd.DataFrame: Average SHAP importance.
    """
    per_fold_shap_importances_combined = pd.concat(per_fold_shap_importances)

    average_shap_importance = (
        per_fold_shap_importances_combined.groupby("feature")
        .agg({"importance": "mean", "percentage": "mean"})
        .reset_index()
    )
    average_shap_importance.columns = ["feature", "importance_mean", "percentage_mean"]
    average_shap_importance = average_shap_importance.sort_values("importance_mean", ascending=False).reset_index(
        drop=True
    )

    logger.info(
        f"Ranked features by average SHAP importance across CV folds\n{average_shap_importance.to_string(index=False)}"
    )

    logger.info("Cumulative importance")
    logger.info(f"  Top 10: {average_shap_importance.head(10)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 20: {average_shap_importance.head(20)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 30: {average_shap_importance.head(30)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 40: {average_shap_importance.head(40)['percentage_mean'].sum():5.1f}%")
    logger.info(f"  Top 50: {average_shap_importance.head(50)['percentage_mean'].sum():5.1f}%")

    return average_shap_importance


def load_feature_selection_results() -> dict[str, pd.DataFrame]:
    """
    Load the feature selection results from all directories.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping experiment names to feature selection results DataFrames.
    """
    required_columns = {"model", "feature", "importance_mean", "percentage_mean"}
    valid_results = {}

    for subdir in OUTPUTS_DIR.iterdir():
        experiment_name = subdir.name
        if not subdir.is_dir() or not experiment_name.startswith("features_selection_"):
            continue

        feature_selection_results_path = subdir / RESULTS_DIRNAME / FEATURE_SELECTION_RESULTS_FILENAME

        if not feature_selection_results_path.exists():
            continue

        df = pd.read_csv(feature_selection_results_path)

        if required_columns.issubset(df.columns):
            valid_results[experiment_name] = df
            logger.info(f"Loaded feature selection results from {experiment_name}")

    logger.info(f"Loaded {len(valid_results)} valid feature selection result files")

    return valid_results


def rank_feature_importance_results(feature_importance_results: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Rank the feature importance results.

    Args:
        feature_importance_results (pd.DataFrame): Feature importance results.
        experiment_name (str): Experiment name.

    Returns:
        pd.DataFrame: Ranked feature importance results.
    """
    result_frames = []

    for _, group in feature_importance_results.groupby("model"):
        group = group.copy()

        group[f"rank_{experiment_name}"] = group["percentage_mean"].rank(ascending=False, method="dense")

        minimum_value = group["percentage_mean"].min()
        maximum_value = group["percentage_mean"].max()
        group[f"scaled_{experiment_name}"] = (group["percentage_mean"] - minimum_value) / (
            maximum_value - minimum_value
        )

        result_frames.append(group)

    return pd.concat(result_frames, ignore_index=True)


def combine_feature_rankings(
    feature_selection_results: dict[str, pd.DataFrame], alpha: float = RANKING_ALPHA
) -> pd.DataFrame:
    """
    Combine multiple feature importance DataFrames into a single aggregated ranking.

    Args:
        feature_selection_results (dict[str, pd.DataFrame]): Dictionary mapping experiment names to feature selection results DataFrames .
        alpha (float): Weight for scaled vs rank in final score.

    Returns:
        pd.DataFrame: Final ranking DataFrame.
    """
    ranked_dfs = []
    for experiment_name, df in feature_selection_results.items():
        ranked_dfs.append(rank_feature_importance_results(df, experiment_name))

    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["model", "feature"], how="outer"), ranked_dfs)

    rank_cols = [column for column in merged_df.columns if column.startswith("rank_")]
    scaled_cols = [column for column in merged_df.columns if column.startswith("scaled_")]
    merged_df["average_rank_model"] = merged_df[rank_cols].mean(axis=1)
    merged_df["average_scaled_model"] = merged_df[scaled_cols].mean(axis=1)

    final_df = (
        merged_df.groupby("feature")
        .agg(final_rank=("average_rank_model", "mean"), final_scaled=("average_scaled_model", "mean"))
        .reset_index()
    )

    minimum_rank, maximum_rank = final_df["final_rank"].min(), final_df["final_rank"].max()
    final_df["normalized_rank"] = (
        0.0 if maximum_rank == minimum_rank else (final_df["final_rank"] - minimum_rank) / (maximum_rank - minimum_rank)
    )

    final_df["final_score"] = alpha * final_df["final_scaled"] + (1 - alpha) * (1 - final_df["normalized_rank"])

    columns = ["feature", "final_score", "final_rank", "final_scaled", "normalized_rank"]
    other_columns = [column for column in final_df.columns if column not in columns]
    final_df = final_df[columns + other_columns]
    final_df = final_df.sort_values("final_score", ascending=False).reset_index(drop=True)

    return final_df


def load_feature_ranking_results() -> pd.DataFrame:
    """
    Load the feature ranking results.

    Returns:
        pd.DataFrame: Feature ranking results.
    """
    feature_ranking_results_path = (
        OUTPUTS_DIR / "features_selection_ranking" / RESULTS_DIRNAME / FEATURE_SELECTION_RESULTS_FILENAME
    )

    if not feature_ranking_results_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(feature_ranking_results_path)

    return df


def compare_correlated_features_pairs(
    correlated_feature_pairs: pd.DataFrame, feature_ranking: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare correlated feature pairs against their ranking scores.

    Args:
        correlated_feature_pairs: DataFrame with correlated feature pairs.
        feature_ranking: DataFrame with feature rankings.

    Returns:
        pd.DataFrame: DataFrame with correlated feature pairs comparison results.
    """
    feature_final_scores = dict(zip(feature_ranking["feature"], feature_ranking["final_score"]))
    feature_index = dict(zip(feature_ranking["feature"], feature_ranking.index + 1))

    rows = []
    for row in correlated_feature_pairs.itertuples(index=False):
        feature_1, feature_2, correlation = row.feature_1, row.feature_2, row.correlation
        feature_1_final_score, feature_1_final_index = (
            feature_final_scores.get(feature_1, -1),
            feature_index.get(feature_1, -1),
        )
        feature_2_final_score, feature_2_final_index = (
            feature_final_scores.get(feature_2, -1),
            feature_index.get(feature_2, -1),
        )

        if feature_1_final_score >= feature_2_final_score:
            keep, keep_index = feature_1, feature_1_final_index
        else:
            keep, keep_index = feature_2, feature_2_final_index

        rows.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_1_score": feature_1_final_score,
                "feature_2_score": feature_2_final_score,
                "feature_1_rank": feature_1_final_index,
                "feature_2_rank": feature_2_final_index,
                "correlation": correlation,
                "keep": keep,
                "keep_index": keep_index,
            }
        )

    return pd.DataFrame(rows)


def add_feature_categories(feature_ranking: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature categories to the feature ranking.

    Args:
        feature_ranking: DataFrame with feature rankings.

    Returns:
        pd.DataFrame: DataFrame with feature categories.
    """
    feature_category_map = {}

    for category, features in asdict(FEATURE_CATEGORIES).items():
        for feature in features:
            feature_category_map[feature] = category

    feature_ranking["category"] = feature_ranking["feature"].map(feature_category_map).fillna("none")
    return feature_ranking
