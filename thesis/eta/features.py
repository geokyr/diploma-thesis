import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def log_transform(
    data: pd.DataFrame | np.ndarray | pd.Series, feature_names: list[str] = ["distance"]
) -> pd.DataFrame | np.ndarray | pd.Series:
    """
    Log transform data, either a DataFrame with features or a NumPy array/pandas Series.

    Args:
        data (pd.DataFrame | np.ndarray | pd.Series): DataFrame or NumPy array/pandas Series to log transform.
        feature_names (list[str]): List of feature names to log transform, only used when data is a DataFrame.

    Returns:
        Data with log-transformed features, same type as input.
    """
    if isinstance(data, pd.DataFrame):
        data_transformed = data.copy()
        data_transformed[feature_names] = np.log(data_transformed[feature_names])
        logger.info(f"Applied log transformation to DataFrame features: {feature_names}")
        return data_transformed
    else:
        data_transformed = np.log(data)
        logger.info("Applied log transformation to NumPy array/pandas Series")
        return data_transformed


def reverse_log_transform(data: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """
    Convert log-transformed data back to original scale.

    Args:
        data (np.ndarray | pd.Series): Data in log space.

    Returns:
        Data in original scale.
    """
    reversed_data = np.exp(data)
    logger.info(f"Reversed log transformation on {len(reversed_data)} samples")
    return reversed_data


def standard_scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard scale training and test features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features to fit scaler on.
        X_test (pd.DataFrame): Test features to transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    logger.info(f"Standard scaled {len(X_train)} training samples and {len(X_test)} test samples")

    return X_train_scaled, X_test_scaled


def engineer_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features from trip data.

    Args:
        df (pd.DataFrame): DataFrame with trip data.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    df_features = df.copy()

    # Basic trip features
    if "distance" in df_features.columns and "duration" in df_features.columns:
        # Average speed in km/h
        df_features["avg_speed_kmh"] = (df_features["distance"] / df_features["duration"]) * 3.6
        logger.info("Added average speed feature")

    # Coordinate-based features
    if all(col in df_features.columns for col in ["origin_x", "origin_y", "destination_x", "destination_y"]):
        # Euclidean distance
        df_features["euclidean_distance"] = np.sqrt(
            (df_features["destination_x"] - df_features["origin_x"]) ** 2
            + (df_features["destination_y"] - df_features["origin_y"]) ** 2
        )

        # Manhattan distance
        df_features["manhattan_distance"] = np.abs(df_features["destination_x"] - df_features["origin_x"]) + np.abs(
            df_features["destination_y"] - df_features["origin_y"]
        )

        # Distance ratio (actual/euclidean) - indicates route directness
        df_features["distance_ratio"] = df_features["distance"] / df_features["euclidean_distance"]

        # Trip direction (angle in radians)
        df_features["trip_angle"] = np.arctan2(
            df_features["destination_y"] - df_features["origin_y"],
            df_features["destination_x"] - df_features["origin_x"],
        )

        # Coordinate differences
        df_features["x_diff"] = df_features["destination_x"] - df_features["origin_x"]
        df_features["y_diff"] = df_features["destination_y"] - df_features["origin_y"]

        # Trip center point
        df_features["center_x"] = (df_features["origin_x"] + df_features["destination_x"]) / 2
        df_features["center_y"] = (df_features["origin_y"] + df_features["destination_y"]) / 2

        logger.info("Added coordinate-based features")

    # Time-based features
    if "hour_bin" in df_features.columns:
        # Cyclical encoding for hour (sin/cos to preserve circular nature)
        df_features["hour_sin"] = np.sin(2 * np.pi * df_features["hour_bin"] / 24)
        df_features["hour_cos"] = np.cos(2 * np.pi * df_features["hour_bin"] / 24)

        # Time of day categories
        df_features["is_morning_rush"] = df_features["hour_bin"].isin([7, 8, 9]).astype(int)
        df_features["is_evening_rush"] = df_features["hour_bin"].isin([17, 18, 19]).astype(int)
        df_features["is_night"] = df_features["hour_bin"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)

        logger.info("Added time-based features")

    return df_features


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
