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
    if "time_start" not in df.columns:
        logger.warning("time_start column not found, skipping hour features")
        return df

    df_hour = df.copy()

    df_hour["hour_bin"] = (df_hour["time_start"] // 3600) % 24
    df_hour["hour_sin"] = np.sin(2 * np.pi * df_hour["hour_bin"] / 24)
    df_hour["hour_cos"] = np.cos(2 * np.pi * df_hour["hour_bin"] / 24)

    n_initial_features = len(df.columns)
    n_final_features = len(df_hour.columns)
    n_added_features = n_final_features - n_initial_features

    logger.info(f"Added {n_added_features} hour features, for a total of {n_final_features} features")

    return df_hour


def add_time_period_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time period features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing hour_bin column.

    Returns:
        pd.DataFrame: DataFrame with added time period features.
    """
    if "hour_bin" not in df.columns:
        logger.warning("hour_bin column not found, skipping time period features")
        return df

    df_time_period = df.copy()

    df_time_period["is_morning"] = (df_time_period["hour_bin"] <= 2).astype(int)
    df_time_period["is_noon"] = ((df_time_period["hour_bin"] >= 3) & (df_time_period["hour_bin"] <= 6)).astype(int)
    df_time_period["is_afternoon"] = (df_time_period["hour_bin"] >= 7).astype(int)
    df_time_period["is_rush_hour"] = df_time_period["hour_bin"].isin([0, 1, 8, 9]).astype(int)

    n_initial_features = len(df.columns)
    n_final_features = len(df_time_period.columns)
    n_added_features = n_final_features - n_initial_features

    logger.info(f"Added {n_added_features} time period features, for a total of {n_final_features} features")

    return df_time_period


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing time_start column.

    Returns:
        pd.DataFrame: DataFrame with added temporal features.
    """
    if "time_start" not in df.columns:
        logger.warning("time_start column not found, skipping temporal features")
        return df

    df_temporal = add_hour_features(df)
    df_temporal = add_time_period_features(df_temporal)

    n_initial_features = len(df.columns)
    n_final_features = len(df_temporal.columns)
    n_added_features = n_final_features - n_initial_features

    logger.info(f"Added {n_added_features} temporal features, for a total of {n_final_features} features")

    return df_temporal


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
    if not all(column in df.columns for column in required_columns):
        missing_columns = [column for column in required_columns if column not in df.columns]
        logger.warning(f"{missing_columns} not found, skipping spatial features")
        return df

    df_spatial = df.copy()

    dx = df_spatial["destination_x"] - df_spatial["source_x"]
    dy = df_spatial["destination_y"] - df_spatial["source_y"]

    df_spatial["euclidean_distance"] = np.hypot(dx, dy)
    df_spatial["manhattan_distance"] = np.abs(dx) + np.abs(dy)
    df_spatial["route_efficiency"] = df_spatial["euclidean_distance"] / df_spatial["distance"]
    df_spatial["route_complexity"] = df_spatial["manhattan_distance"] / df_spatial["euclidean_distance"]
    df_spatial["detour_factor"] = df_spatial["distance"] - df_spatial["euclidean_distance"]

    df_spatial["trip_bearing"] = np.arctan2(dy, dx)
    df_spatial["trip_bearing_sin"] = np.sin(df_spatial["trip_bearing"])
    df_spatial["trip_bearing_cos"] = np.cos(df_spatial["trip_bearing"])
    df_spatial["x_difference"] = dx
    df_spatial["y_difference"] = dy
    df_spatial["x_center"] = (df_spatial["source_x"] + df_spatial["destination_x"]) / 2
    df_spatial["y_center"] = (df_spatial["source_y"] + df_spatial["destination_y"]) / 2

    distance_percentiles = np.percentile(df_spatial["distance"], [25, 50, 75])

    df_spatial["is_short_distance"] = (df_spatial["distance"] <= distance_percentiles[0]).astype(int)
    df_spatial["is_medium_distance"] = (
        (df_spatial["distance"] > distance_percentiles[0]) & (df_spatial["distance"] <= distance_percentiles[2])
    ).astype(int)
    df_spatial["is_long_distance"] = (df_spatial["distance"] > distance_percentiles[2]).astype(int)

    coordinate_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    for column in coordinate_columns:
        normalized_coordinates = df_spatial[column] / coordinate_scale

        freqs = np.array([2**i for i in range(num_freqs)])

        for i, freq in enumerate(freqs):
            df_spatial[f"{column}_sin_freq_{i}"] = np.sin(freq * normalized_coordinates)
            df_spatial[f"{column}_cos_freq_{i}"] = np.cos(freq * normalized_coordinates)

    df_spatial["source_cell_x"] = (df_spatial["source_x"] // cell).astype(int)
    df_spatial["source_cell_y"] = (df_spatial["source_y"] // cell).astype(int)
    df_spatial["destination_cell_x"] = (df_spatial["destination_x"] // cell).astype(int)
    df_spatial["destination_cell_y"] = (df_spatial["destination_y"] // cell).astype(int)

    source_coordinates = df_spatial[["source_x", "source_y"]].values
    destination_coordinates = df_spatial[["destination_x", "destination_y"]].values
    kmeans_source = KMeans(n_clusters=n_clusters, random_state=random_seed)
    kmeans_destination = KMeans(n_clusters=n_clusters, random_state=random_seed)

    df_spatial["source_cluster"] = kmeans_source.fit_predict(source_coordinates)
    df_spatial["destination_cluster"] = kmeans_destination.fit_predict(destination_coordinates)

    all_coordinates = np.column_stack(
        [df_spatial["source_x"], df_spatial["source_y"], df_spatial["destination_x"], df_spatial["destination_y"]]
    )

    pca_coordinates = PCA(n_components=2, random_state=random_seed)
    coordinates_pca = pca_coordinates.fit_transform(all_coordinates)

    trip_vectors = np.column_stack([dx, dy])
    pca_trip_vectors = PCA(n_components=2, random_state=random_seed)
    trip_vectors_pca = pca_trip_vectors.fit_transform(trip_vectors)

    df_spatial["coordinates_pca_1"] = coordinates_pca[:, 0]
    df_spatial["coordinates_pca_2"] = coordinates_pca[:, 1]
    df_spatial["trip_vectors_pca_1"] = trip_vectors_pca[:, 0]
    df_spatial["trip_vectors_pca_2"] = trip_vectors_pca[:, 1]

    df_spatial["distance_from_center"] = np.hypot(df_spatial["x_center"], df_spatial["y_center"])
    df_spatial["source_distance_from_center"] = np.hypot(df_spatial["source_x"], df_spatial["source_y"])
    df_spatial["dest_distance_from_center"] = np.hypot(df_spatial["destination_x"], df_spatial["destination_y"])
    df_spatial["trip_towards_center"] = (
        df_spatial["source_distance_from_center"] > df_spatial["dest_distance_from_center"]
    ).astype(int)
    df_spatial["trip_away_from_center"] = (
        df_spatial["source_distance_from_center"] < df_spatial["dest_distance_from_center"]
    ).astype(int)

    num_of_initial_features = len(df.columns)
    num_of_final_features = len(df_spatial.columns)
    num_of_added_features = num_of_final_features - num_of_initial_features

    logger.info(f"Added {num_of_added_features} spatial features, for a total of {num_of_final_features} features")

    return df_spatial
