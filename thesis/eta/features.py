"""
Feature engineering and transformation utilities for ETA prediction.
Provides functions for scaling, transforming features and targets, engineering spatial and temporal features from trip data.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler

from thesis.common.config import RANDOM_SEED_DEFAULT

logger = logging.getLogger(__name__)


def _check_required_columns(df: pd.DataFrame, required_columns: list[str], feature_type: str) -> bool:
    """
    Check if DataFrame contains all required columns.

    Args:
        df (pd.DataFrame): DataFrame to check.
        required_columns (list[str]): List of required column names.
        feature_type (str): Type of features being added (for logging).

    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]
        logger.warning(f"{missing_columns} not found, skipping {feature_type} features")
        return False
    return True


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
    df: pd.DataFrame, target_columns: list[str] = ["duration"]
) -> tuple[pd.DataFrame, pd.DataFrame | pd.Series]:
    """
    Split the features and the target.

    Args:
        df (pd.DataFrame): A DataFrame containing the prepared data.
        target_columns (list[str]): List of column names to use as target, defaults to ["duration"].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame | pd.Series]: A tuple containing the features and the target.

    Raises:
        ValueError: If the DataFrame does not contain the target columns.
    """
    missing_columns = [col for col in target_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain target columns: {missing_columns}")

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


def create_quantile_normal_transformer(random_seed: int = RANDOM_SEED_DEFAULT) -> QuantileTransformer:
    """
    Create a quantile normal transformer.

    Args:
        random_seed (int): The random seed to use for the random number generator.

    Returns:
        QuantileTransformer: Quantile normal transformer.
    """
    return QuantileTransformer(output_distribution="normal", random_state=random_seed)


def create_box_cox_transformer() -> PowerTransformer:
    """
    Create a box cox transformer.

    Returns:
        PowerTransformer: Box cox transformer.
    """
    return PowerTransformer(method="box-cox")


def add_hour_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing time_start column.

    Returns:
        pd.DataFrame: DataFrame with added hour features.
    """
    required_columns = ["time_start"]
    if not _check_required_columns(df, required_columns, "hour"):
        return df

    df_hour = df.copy()

    df_hour["hour_bin"] = (df_hour["time_start"] // 3600) % 24
    df_hour["hour_sin"] = np.sin(2 * np.pi * df_hour["hour_bin"] / 24)
    df_hour["hour_cos"] = np.cos(2 * np.pi * df_hour["hour_bin"] / 24)

    _log_feature_addition(df, df_hour, "hour")

    return df_hour


def add_time_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time period features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing hour_bin column.

    Returns:
        pd.DataFrame: DataFrame with added time period features.
    """
    required_columns = ["hour_bin"]
    if not _check_required_columns(df, required_columns, "time period"):
        return df

    df_time_period = df.copy()

    df_time_period["is_morning"] = (df_time_period["hour_bin"] <= 2).astype(int)
    df_time_period["is_noon"] = ((df_time_period["hour_bin"] >= 3) & (df_time_period["hour_bin"] <= 6)).astype(int)
    df_time_period["is_afternoon"] = (df_time_period["hour_bin"] >= 7).astype(int)
    df_time_period["is_rush_hour"] = df_time_period["hour_bin"].isin([0, 1, 8, 9]).astype(int)

    _log_feature_addition(df, df_time_period, "time period")

    return df_time_period


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing time_start column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    required_columns = ["time_start"]
    if not _check_required_columns(df, required_columns, "temporal"):
        return df

    df_temporal = add_hour_features(df)
    df_temporal = add_time_period_features(df_temporal)

    _log_feature_addition(df, df_temporal, "temporal")

    return df_temporal


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coordinate features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.

    Returns:
        pd.DataFrame: DataFrame with added coordinate features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "coordinate"):
        return df

    df_coordinate = df.copy()

    dx = df_coordinate["destination_x"] - df_coordinate["source_x"]
    dy = df_coordinate["destination_y"] - df_coordinate["source_y"]

    df_coordinate["euclidean_distance"] = np.hypot(dx, dy)
    df_coordinate["manhattan_distance"] = np.abs(dx) + np.abs(dy)

    _log_feature_addition(df, df_coordinate, "coordinate")

    return df_coordinate


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distance features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing euclidean_distance, manhattan_distance, distance columns.

    Returns:
        pd.DataFrame: DataFrame with added distance features.
    """
    required_columns = ["euclidean_distance", "manhattan_distance", "distance"]
    if not _check_required_columns(df, required_columns, "distance"):
        return df

    df_distance = df.copy()

    df_distance["route_efficiency"] = df_distance["euclidean_distance"] / df_distance["distance"]
    df_distance["route_complexity"] = df_distance["manhattan_distance"] / df_distance["euclidean_distance"]
    df_distance["detour_factor"] = df_distance["distance"] - df_distance["euclidean_distance"]

    distance_percentiles = np.percentile(df_distance["distance"], [25, 50, 75])

    df_distance["is_short_distance"] = (df_distance["distance"] <= distance_percentiles[0]).astype(int)
    df_distance["is_medium_distance"] = (
        (df_distance["distance"] > distance_percentiles[0]) & (df_distance["distance"] <= distance_percentiles[2])
    ).astype(int)
    df_distance["is_long_distance"] = (df_distance["distance"] > distance_percentiles[2]).astype(int)

    _log_feature_addition(df, df_distance, "distance")

    return df_distance


def add_trip_vector_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trip vector features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.

    Returns:
        pd.DataFrame: DataFrame with added trip vector features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "trip vector"):
        return df

    df_vector = df.copy()

    df_vector["x_difference"] = df_vector["destination_x"] - df_vector["source_x"]
    df_vector["y_difference"] = df_vector["destination_y"] - df_vector["source_y"]
    df_vector["trip_bearing"] = np.arctan2(df_vector["y_difference"], df_vector["x_difference"])
    df_vector["trip_bearing_sin"] = np.sin(df_vector["trip_bearing"])
    df_vector["trip_bearing_cos"] = np.cos(df_vector["trip_bearing"])
    df_vector["x_center"] = (df_vector["source_x"] + df_vector["destination_x"]) / 2
    df_vector["y_center"] = (df_vector["source_y"] + df_vector["destination_y"]) / 2

    _log_feature_addition(df, df_vector, "trip vector")

    return df_vector


def add_fourier_features(df: pd.DataFrame, num_freqs: int = 2, coordinate_scale: float = 1000.0) -> pd.DataFrame:
    """
    Add Fourier positional encoding features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        num_freqs (int): Number of frequency components to use for Fourier encoding.
        coordinate_scale (float): Scale factor to normalize coordinates for Fourier encoding.

    Returns:
        pd.DataFrame: DataFrame with added Fourier features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "Fourier"):
        return df

    df_fourier = df.copy()

    coordinate_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    for column in coordinate_columns:
        normalized_coordinates = df_fourier[column] / coordinate_scale

        freqs = np.array([2**i for i in range(num_freqs)])

        for i, freq in enumerate(freqs):
            df_fourier[f"{column}_sin_freq_{i}"] = np.sin(freq * normalized_coordinates)
            df_fourier[f"{column}_cos_freq_{i}"] = np.cos(freq * normalized_coordinates)

    _log_feature_addition(df, df_fourier, "Fourier")

    return df_fourier


def add_cell_features(df: pd.DataFrame, cell: int = 100) -> pd.DataFrame:
    """
    Add cell features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        cell (int): Size of the cell in meters.

    Returns:
        pd.DataFrame: DataFrame with added cell features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "cell"):
        return df

    df_cell = df.copy()

    df_cell["source_cell_x"] = (df_cell["source_x"] // cell).astype(int)
    df_cell["source_cell_y"] = (df_cell["source_y"] // cell).astype(int)
    df_cell["destination_cell_x"] = (df_cell["destination_x"] // cell).astype(int)
    df_cell["destination_cell_y"] = (df_cell["destination_y"] // cell).astype(int)

    _log_feature_addition(df, df_cell, "cell")

    return df_cell


def add_clustering_features(
    df: pd.DataFrame, n_clusters: int = 20, random_seed: int = RANDOM_SEED_DEFAULT
) -> pd.DataFrame:
    """
    Add clustering features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering.

    Returns:
        pd.DataFrame: DataFrame with added clustering features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "clustering"):
        return df

    df_cluster = df.copy()

    source_coordinates = df_cluster[["source_x", "source_y"]].values
    destination_coordinates = df_cluster[["destination_x", "destination_y"]].values
    kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)

    df_cluster["source_cluster"] = kmeans_source.fit_predict(source_coordinates)
    df_cluster["destination_cluster"] = kmeans_destination.fit_predict(destination_coordinates)

    _log_feature_addition(df, df_cluster, "clustering")

    return df_cluster


def add_pca_features(df: pd.DataFrame, random_seed: int = RANDOM_SEED_DEFAULT) -> pd.DataFrame:
    """
    Add PCA features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        random_seed (int): Random seed for PCA.

    Returns:
        pd.DataFrame: DataFrame with added PCA features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "PCA"):
        return df

    df_pca = df.copy()

    dx = df_pca["destination_x"] - df_pca["source_x"]
    dy = df_pca["destination_y"] - df_pca["source_y"]

    all_coordinates = np.column_stack(
        [df_pca["source_x"], df_pca["source_y"], df_pca["destination_x"], df_pca["destination_y"]]
    )

    pca_coordinates = PCA(n_components=2, random_state=random_seed)
    coordinates_pca = pca_coordinates.fit_transform(all_coordinates)

    trip_vectors = np.column_stack([dx, dy])
    pca_trip_vectors = PCA(n_components=2, random_state=random_seed)
    trip_vectors_pca = pca_trip_vectors.fit_transform(trip_vectors)

    df_pca["coordinates_pca_1"] = coordinates_pca[:, 0]
    df_pca["coordinates_pca_2"] = coordinates_pca[:, 1]
    df_pca["trip_vectors_pca_1"] = trip_vectors_pca[:, 0]
    df_pca["trip_vectors_pca_2"] = trip_vectors_pca[:, 1]

    _log_feature_addition(df, df_pca, "PCA")

    return df_pca


def add_center_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add center distance features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, x_center, y_center columns.

    Returns:
        pd.DataFrame: DataFrame with added center distance features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y", "x_center", "y_center"]
    if not _check_required_columns(df, required_columns, "center distance"):
        return df

    df_center = df.copy()

    df_center["distance_from_center"] = np.hypot(df_center["x_center"], df_center["y_center"])
    df_center["source_distance_from_center"] = np.hypot(df_center["source_x"], df_center["source_y"])
    df_center["dest_distance_from_center"] = np.hypot(df_center["destination_x"], df_center["destination_y"])
    df_center["trip_towards_center"] = (
        df_center["source_distance_from_center"] > df_center["dest_distance_from_center"]
    ).astype(int)
    df_center["trip_away_from_center"] = (
        df_center["source_distance_from_center"] < df_center["dest_distance_from_center"]
    ).astype(int)

    _log_feature_addition(df, df_center, "center distance")

    return df_center


def add_spatial_features(
    df: pd.DataFrame,
    num_freqs: int = 2,
    coordinate_scale: float = 1000.0,
    cell: int = 100,
    n_clusters: int = 20,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add spatial features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        num_freqs (int): Number of frequency components to use for Fourier encoding.
        coordinate_scale (float): Scale factor to normalize coordinates for Fourier encoding.
        cell (int): Size of the cell in meters.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering and PCA.

    Returns:
        pd.DataFrame: DataFrame with added spatial features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
    if not _check_required_columns(df, required_columns, "spatial"):
        return df

    df_spatial = add_coordinate_features(df)
    df_spatial = add_distance_features(df_spatial)
    df_spatial = add_trip_vector_features(df_spatial)
    df_spatial = add_fourier_features(df_spatial, num_freqs, coordinate_scale)
    df_spatial = add_cell_features(df_spatial, cell)
    df_spatial = add_clustering_features(df_spatial, n_clusters, random_seed)
    df_spatial = add_pca_features(df_spatial, random_seed)
    df_spatial = add_center_distance_features(df_spatial)

    _log_feature_addition(df, df_spatial, "spatial")

    return df_spatial


def add_all_features(
    df: pd.DataFrame,
    num_freqs: int = 2,
    coordinate_scale: float = 1000.0,
    cell: int = 100,
    n_clusters: int = 20,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> pd.DataFrame:
    """
    Add all features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.
        num_freqs (int): Number of frequency components to use for Fourier encoding.
        coordinate_scale (float): Scale factor to normalize coordinates for Fourier encoding.
        cell (int): Size of the cell in meters.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering and PCA.

    Returns:
        pd.DataFrame: DataFrame with added all features.
    """
    required_columns = ["time_start", "source_x", "source_y", "destination_x", "destination_y", "distance"]
    if not _check_required_columns(df, required_columns, "all"):
        return df

    df_all = df.copy()

    df_all = add_temporal_features(df_all)
    df_all = add_spatial_features(df_all, num_freqs, coordinate_scale, cell, n_clusters, random_seed)

    _log_feature_addition(df, df_all, "all")

    return df_all
