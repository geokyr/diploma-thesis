import pandas as pd

from thesis.logger import ETA_LOGGER_NAME, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME)


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
    logger.info("Splitting features and target...")

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
    logger.info(f"Split {n_samples} samples into {len(X.columns)} features and {n_targets} target {variable_word}.")
    return X, y
