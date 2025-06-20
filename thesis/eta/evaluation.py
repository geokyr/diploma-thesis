import logging
import time

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def make_predictions(
    model: BaseEstimator,
    model_name: str,
    X_test: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    """
    Make predictions using a trained model.

    Args:
        model (BaseEstimator): The machine learning model to use for predictions.
        model_name (str): The name of the model.
        X_test (pd.DataFrame): A DataFrame containing the test features.

    Returns:
        tuple[pd.Series, dict[str, float]]: A tuple containing the predictions and timing metrics.
    """
    logger.info(f"Making predictions with {model_name}...")

    prediction_start = time.perf_counter()
    preds = model.predict(X_test)
    prediction_end = time.perf_counter()
    prediction_time = prediction_end - prediction_start

    logger.info(f"{model_name} - Prediction: {prediction_time:.3f}s")

    return pd.Series(preds), {"prediction": prediction_time}


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str,
) -> dict[str, float]:
    """
    Evaluate predictions against true values.

    Args:
        y_true (pd.Series): A Series containing the true target values.
        y_pred (pd.Series): A Series containing the predicted target values.
        model_name (str): The name of the model (for logging).

    Returns:
        dict[str, float]: A dictionary containing evaluation metrics.
    """
    logger.info(f"Evaluating predictions for {model_name}...")

    evaluation_start = time.perf_counter()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evaluation_end = time.perf_counter()
    evaluation_time = evaluation_end - evaluation_start

    logger.info(
        f"{model_name} - Evaluation: {evaluation_time:.3f}s, MAE: {mae:.2f}s, MSE: {mse:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape * 100:.2f}%, R2: {r2:.3f}"
    )

    return {
        "evaluation": evaluation_time,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }
