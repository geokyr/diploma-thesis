import logging
import time

import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


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
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_end = time.perf_counter()
    training_time = training_end - training_start

    logger.info(f"{model_name} - Training: {training_time:.3f}s")

    return {"training": training_time}
