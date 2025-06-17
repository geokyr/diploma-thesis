import time

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from thesis.logger import ETA_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME, log_file=LOG_FILES_CONFIG[ETA_LOGGER_NAME])


def evaluate_model(
    model: BaseEstimator,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float, float, float]:
    """
    Evaluate a model.

    Args:
        model (BaseEstimator): The machine learning model to evaluate.
        model_name (str): The name of the model.
        X_test (pd.DataFrame): A DataFrame containing the test features.
        y_test (pd.Series): A Series containing the test target variable.

    Returns:
        tuple[float, float, float, float]: A tuple that has the evaluation time, MAE, RMSE, and MAPE.
    """
    logger.info(f"Evaluating {model_name}...")

    evaluation_start = time.perf_counter()
    preds = model.predict(X_test)
    evaluation_end = time.perf_counter()
    evaluation_time = evaluation_end - evaluation_start

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    logger.info(
        f"{model_name} - Evaluation: {evaluation_time:.3f}s, MAE: {mae:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape * 100:.2f}%"
    )

    return evaluation_time, mae, rmse, mape
