import logging

import numpy as np
import pandas as pd

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
    return np.exp(data)


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
    logger.info("Splitting features and target")

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
