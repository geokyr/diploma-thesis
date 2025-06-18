import time

import pandas as pd
from sklearn.base import BaseEstimator

from thesis.logger import ETA_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME, log_file=LOG_FILES_CONFIG[ETA_LOGGER_NAME])


def train_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """
    Train a model.

    Args:
        model (BaseEstimator): The machine learning model to train.
        model_name (str): The name of the model.
        X_train (pd.DataFrame): A DataFrame containing the training features.
        y_train (pd.Series): A Series containing the training target variable.

    Returns:
        dict[str, float]: A dictionary containing training metrics.
    """
    logger.info(f"Training {model_name}...")

    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_end = time.perf_counter()
    training_time = training_end - training_start

    logger.info(f"{model_name} - Training: {training_time:.3f}s")

    return {"training": training_time}
