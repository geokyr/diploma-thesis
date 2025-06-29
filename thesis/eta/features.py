import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def log_transform_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, feature_names: list[str] = ["distance"]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Log transform features.

    Args:
        X_train (pd.DataFrame): Training features to log transform.
        X_test (pd.DataFrame): Test features to log transform.
        feature_names (list[str]): List of feature names to log transform.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Log transformed training and test features.
    """
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()

    X_train_log[feature_names] = np.log1p(X_train[feature_names])
    X_test_log[feature_names] = np.log1p(X_test[feature_names])

    logger.info(f"Log transformed {len(X_train)} training samples and {len(X_test)} test samples")
    return X_train_log, X_test_log


def create_log_transformer() -> FunctionTransformer:
    """
    Create a log transformer with exponential inverse function.

    Returns:
        FunctionTransformer: Log transformer with exponential inverse function.
    """
    return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=True)


def analyze_target_distribution(y: pd.Series) -> dict:
    """
    Analyze target distribution to help decide on transformation.

    Args:
        y (pd.Series): Target variable to analyze.

    Returns:
        dict: Statistics about the target distribution.
    """
    from scipy import stats

    stats_dict = {
        "mean": y.mean(),
        "median": y.median(),
        "std": y.std(),
        "min": y.min(),
        "max": y.max(),
        "skewness": stats.skew(y),
        "kurtosis": stats.kurtosis(y),
        "has_zeros": (y == 0).any(),
        "has_negative": (y < 0).any(),
        "pct_zeros": (y == 0).sum() / len(y) * 100,
        "pct_negative": (y < 0).sum() / len(y) * 100,
    }

    # Test for normality
    if len(y) > 20:
        _, p_value = stats.normaltest(y)
        stats_dict["normality_p_value"] = p_value
        stats_dict["is_normal"] = p_value > 0.05

    logger.info(
        f"Target distribution - Skewness: {stats_dict['skewness']:.3f}, "
        f"Has zeros: {stats_dict['has_zeros']}, "
        f"Has negative: {stats_dict['has_negative']}"
    )

    return stats_dict


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
