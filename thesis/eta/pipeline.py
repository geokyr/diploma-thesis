import logging

import pandas as pd
from sklearn.base import BaseEstimator

from thesis.eta.evaluation import evaluate_predictions, make_predictions
from thesis.eta.training import train_model

logger = logging.getLogger(__name__)


def train_and_evaluate_model(
    model: BaseEstimator,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Train and evaluate a single model on given data.

    Args:
        model (BaseEstimator): The model instance to train and evaluate.
        model_name (str): Name of the model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test targets.

    Returns:
        dict[str, float]: Dictionary containing all results for this model.
    """
    logger.info(f"Training and evaluating {model_name}")

    training_results = train_model(model, model_name, X_train, y_train)
    predictions, prediction_results = make_predictions(model, model_name, X_test)
    evaluation_results = evaluate_predictions(y_test, predictions, model_name)
    model_results = {**training_results, **prediction_results, **evaluation_results}

    logger.info(f"Completed training and evaluation of {model_name}")
    return model_results
