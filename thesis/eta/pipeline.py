"""ETA model training, evaluation, and cross-validation pipeline utilities."""

import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import StratifiedKFold

from thesis.common.config import N_BINS, N_SPLITS, RANDOM_SEED_DEFAULT
from thesis.eta.models import ModelType
from thesis.eta.results import EvaluationResults, PredictionResults, TrainingResults

logger = logging.getLogger(__name__)


def get_stratified_kfold_cv(
    y: pd.Series, n_bins: int = N_BINS, n_splits: int = N_SPLITS, random_seed: int = RANDOM_SEED_DEFAULT
) -> tuple[StratifiedKFold, np.ndarray]:
    """
    Discretize a continuous target variable into quantile bins and return a StratifiedKFold object with a key.

    Args:
        y (pd.Series): Series containing the target variable.
        n_bins (int): Number of bins to discretize the target variable into.
        n_splits (int): Number of splits for cross-validation.
        random_seed (int): Random seed to use for the random number generator.

    Returns:
        tuple[StratifiedKFold, np.ndarray]: Tuple containing the StratifiedKFold object and the key.
    """
    target_bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    stratify_key = target_bins.astype(str)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )

    return skf, stratify_key


def train_model(
    model: BaseEstimator, model_type: ModelType, X_train: pd.DataFrame, y_train: pd.Series, **fit_kwargs
) -> TrainingResults:
    """
    Train a model.

    Args:
        model (BaseEstimator): Machine learning model to train.
        model_type (ModelType): Type of the model.
        X_train (pd.DataFrame): DataFrame containing the training features.
        y_train (pd.Series): Series containing the training target variable.
        **fit_kwargs: Additional keyword arguments to pass to the model's fit method.

    Returns:
        TrainingResults: Training metrics.
    """
    training_start = time.perf_counter()
    model.fit(X_train, y_train, **fit_kwargs)
    training_end = time.perf_counter()
    training_time = training_end - training_start

    logger.info(f"{model_type} - Training time: {training_time:.3f}s")

    return TrainingResults(training_time=training_time)


def make_predictions(
    model: BaseEstimator, model_type: ModelType, X_test: pd.DataFrame
) -> tuple[pd.Series, PredictionResults]:
    """
    Make predictions using a trained model.

    Args:
        model (BaseEstimator): Machine learning model to use for predictions.
        model_type (ModelType): Type of the model.
        X_test (pd.DataFrame): DataFrame containing the test features.

    Returns:
        tuple[pd.Series, PredictionResults]: Tuple containing the predictions and timing metrics.
    """
    prediction_start = time.perf_counter()
    preds = model.predict(X_test)
    prediction_end = time.perf_counter()
    prediction_time = prediction_end - prediction_start

    logger.info(f"{model_type} - Prediction time: {prediction_time:.3f}s")

    return pd.Series(preds), PredictionResults(prediction_time=prediction_time)


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, model_type: ModelType) -> EvaluationResults:
    """
    Evaluate predictions against true values.

    Args:
        y_true (pd.Series): Series containing the true target values.
        y_pred (pd.Series): Series containing the predicted target values.
        model_type (ModelType): Type of the model.

    Returns:
        EvaluationResults: Evaluation metrics.
    """
    evaluation_start = time.perf_counter()
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evaluation_end = time.perf_counter()
    evaluation_time = evaluation_end - evaluation_start

    logger.info(f"{model_type} - Evaluation time: {evaluation_time:.3f}s, MAE: {mae:.2f}s, MAPE: {mape * 100:.2f}%")

    return EvaluationResults(
        evaluation_time=evaluation_time,
        mae=mae,
        mape=mape,
    )


def compute_absolute_errors(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Compute absolute errors between true and predicted values.

    Args:
        y_true (pd.Series): Series containing the true target values.
        y_pred (pd.Series): Series containing the predicted target values.

    Returns:
        pd.Series: Series containing the absolute errors.
    """
    return np.abs(y_true - y_pred)
