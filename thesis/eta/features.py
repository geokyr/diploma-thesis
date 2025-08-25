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
    """
    required_columns = ["time_start"]
    if not _check_required_columns(df, required_columns, "temporal"):
        return df

    df_hour = df.copy()

    df_hour["hour_bin"] = (df_hour["time_start"] // 3600) % 24
    df_hour["is_morning"] = (df_hour["hour_bin"] <= 2).astype(int)
    df_hour["is_noon"] = ((df_hour["hour_bin"] >= 3) & (df_hour["hour_bin"] <= 6)).astype(int)
    df_hour["is_afternoon"] = (df_hour["hour_bin"] >= 7).astype(int)
    df_hour["is_rush_hour"] = df_hour["hour_bin"].isin([0, 1, 8, 9]).astype(int)

    _log_feature_addition(df, df_hour, "temporal")

    return df_hour


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spatial features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y, distance columns.

    Returns:
        pd.DataFrame: DataFrame with added spatial features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y", "distance"]
    if not _check_required_columns(df, required_columns, "spatial"):
        return df

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

    distance_percentiles = np.percentile(df_spatial["distance"], [25, 50, 75])

    df_spatial["is_short_distance"] = (df_spatial["distance"] <= distance_percentiles[0]).astype(int)
    df_spatial["is_medium_distance"] = (
        (df_spatial["distance"] > distance_percentiles[0]) & (df_spatial["distance"] <= distance_percentiles[2])
    ).astype(int)
    df_spatial["is_long_distance"] = (df_spatial["distance"] > distance_percentiles[2]).astype(int)

    city_center_x = np.mean(df_spatial["x_center"])
    city_center_y = np.mean(df_spatial["y_center"])

    df_spatial["source_distance_from_city_center"] = np.hypot(
        df_spatial["source_x"] - city_center_x, df_spatial["source_y"] - city_center_y
    )
    df_spatial["dest_distance_from_city_center"] = np.hypot(
        df_spatial["destination_x"] - city_center_x, df_spatial["destination_y"] - city_center_y
    )
    df_spatial["trip_centrality_change"] = (
        df_spatial["dest_distance_from_city_center"] - df_spatial["source_distance_from_city_center"]
    )
    df_spatial["trip_centrality"] = np.hypot(
        df_spatial["x_center"] - city_center_x, df_spatial["y_center"] - city_center_y
    )

    _log_feature_addition(df, df_spatial, "spatial")

    return df_spatial


def add_fourier_features(df: pd.DataFrame, num_freqs: int = 2, coordinate_scale: float = 1000.0) -> pd.DataFrame:
    """
    Add fourier positional encoding features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        num_freqs (int): Number of frequency components to use for fourier encoding.
        coordinate_scale (float): Scale factor to normalize coordinates for fourier encoding.

    Returns:
        pd.DataFrame: DataFrame with added fourier features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "fourier"):
        return df

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
    Add pca features to the dataframe.

    Args:
        df (pd.DataFrame): DataFrame with trip data containing source_x, source_y, destination_x, destination_y columns.
        random_seed (int): Random seed for pca.

    Returns:
        pd.DataFrame: DataFrame with added pca features.
    """
    required_columns = ["source_x", "source_y", "destination_x", "destination_y"]
    if not _check_required_columns(df, required_columns, "pca"):
        return df

    df_pca = df.copy()

    all_coordinates = np.vstack(
        [df_pca[["source_x", "source_y"]].values, df_pca[["destination_x", "destination_y"]].values]
    )

    pca_coordinates = PCA(n_components=2, random_state=random_seed)
    pca_coordinates.fit(all_coordinates)

    source_coords_pca = pca_coordinates.transform(df_pca[["source_x", "source_y"]].values)
    dest_coords_pca = pca_coordinates.transform(df_pca[["destination_x", "destination_y"]].values)

    df_pca["source_pca_1"] = source_coords_pca[:, 0]
    df_pca["source_pca_2"] = source_coords_pca[:, 1]
    df_pca["dest_pca_1"] = dest_coords_pca[:, 0]
    df_pca["dest_pca_2"] = dest_coords_pca[:, 1]

    _log_feature_addition(df, df_pca, "pca")

    return df_pca


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
        num_freqs (int): Number of frequency components to use for fourier encoding.
        coordinate_scale (float): Scale factor to normalize coordinates for fourier encoding.
        cell (int): Size of the cell in meters.
        n_clusters (int): Number of clusters for K-means clustering on coordinates.
        random_seed (int): Random seed for clustering and pca.

    Returns:
        pd.DataFrame: DataFrame with added all features.
    """
    required_columns = ["time_start", "source_x", "source_y", "destination_x", "destination_y", "distance"]
    if not _check_required_columns(df, required_columns, "all"):
        return df

    df_all = df.copy()

    df_all = add_temporal_features(df_all)
    df_all = add_spatial_features(df_all)
    df_all = add_fourier_features(df_all, num_freqs, coordinate_scale)
    df_all = add_cell_features(df_all, cell)
    df_all = add_clustering_features(df_all, n_clusters, random_seed)
    df_all = add_pca_features(df_all, random_seed)

    _log_feature_addition(df, df_all, "all")

    return df_all
