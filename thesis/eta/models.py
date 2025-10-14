"""Machine learning model definitions and factory functions for ETA prediction."""

import inspect
import json
import logging
from pathlib import Path
from typing import Callable

import joblib
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer
from xgboost import XGBRegressor

from thesis.common.config import (
    ALLOW_WRITING_FILES,
    APPDATA_DIRNAME,
    COLSAMPLE_BYTREE,
    DEFAULT_VERSION,
    ENABLE_CATEGORICAL,
    END_TIME,
    IMPORTANCE_TYPE,
    LEARNING_RATE,
    LOSS_FUNCTION_CATBOOST,
    MAX_DEPTH,
    METADATA_FILENAME,
    MODEL_FILENAME,
    MODELS_DIRNAME,
    N_ESTIMATORS,
    OBJECTIVE_LIGHTGBM,
    OBJECTIVE_XGBOOST,
    PROJECT_DIR,
    RANDOM_SEED_DEFAULT,
    START_TIME,
    SUBSAMPLE,
    TREE_METHOD,
    VERBOSE_CATBOOST,
    VERBOSE_LIGHTGBM,
)
from thesis.common.enums import MLTask, ModelType

logger = logging.getLogger(__name__)


MODEL_REGISTRY: dict[ModelType, Callable[..., BaseEstimator]] = {}


def register_model(model_type: ModelType) -> Callable[[Callable[..., BaseEstimator]], Callable[..., BaseEstimator]]:
    """
    Decorator to register a model creator function.

    Args:
        model_type: The type of model to register.

    Returns:
        Callable[[Callable[..., BaseEstimator]], Callable[..., BaseEstimator]]: The decorator function.
    """

    def decorator(func: Callable[..., BaseEstimator]) -> Callable[..., BaseEstimator]:
        MODEL_REGISTRY[model_type] = func
        return func

    return decorator


@register_model(ModelType.LINEAR_REGRESSION)
def create_linear_regression_model(**kwargs) -> LinearRegression:
    """
    Create a Linear Regression model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LinearRegression: Configured Linear Regression model.
    """
    return LinearRegression(**kwargs)


@register_model(ModelType.XGBOOST_REGRESSOR)
def create_xgboost_regressor_model(random_seed: int = RANDOM_SEED_DEFAULT, **kwargs) -> XGBRegressor:
    """
    Create an XGBoost Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        XGBRegressor: Configured XGBoost Regressor model.
    """
    params = {
        "objective": OBJECTIVE_XGBOOST,
        "random_state": random_seed,
        "enable_categorical": ENABLE_CATEGORICAL,
        "tree_method": TREE_METHOD,
        "importance_type": IMPORTANCE_TYPE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        **kwargs,
    }

    return XGBRegressor(**params)


@register_model(ModelType.LIGHTGBM_REGRESSOR)
def create_lightgbm_regressor_model(random_seed: int = RANDOM_SEED_DEFAULT, **kwargs) -> LGBMRegressor:
    """
    Create a LightGBM Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LGBMRegressor: Configured LightGBM Regressor model.
    """
    params = {
        "objective": OBJECTIVE_LIGHTGBM,
        "random_state": random_seed,
        "verbose": VERBOSE_LIGHTGBM,
        "importance_type": IMPORTANCE_TYPE,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        **kwargs,
    }

    return LGBMRegressor(**params)


@register_model(ModelType.CATBOOST_REGRESSOR)
def create_catboost_regressor_model(random_seed: int = RANDOM_SEED_DEFAULT, **kwargs) -> CatBoostRegressor:
    """
    Create a CatBoost Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        CatBoostRegressor: Configured CatBoost Regressor model.
    """
    params = {
        "loss_function": LOSS_FUNCTION_CATBOOST,
        "random_state": random_seed,
        "verbose": VERBOSE_CATBOOST,
        "allow_writing_files": ALLOW_WRITING_FILES,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bylevel": COLSAMPLE_BYTREE,
        **kwargs,
    }

    return CatBoostRegressor(**params)


def create_model(model_type: ModelType, random_seed: int = RANDOM_SEED_DEFAULT, **kwargs) -> BaseEstimator:
    """
    Create a model by its type.

    Args:
        model_type: The type of model to create.
        random_seed: The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        BaseEstimator: The configured model instance.

    Raises:
        ValueError: If the model type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(str(t) for t in ModelType)
        error_msg = f"Unknown model: {model_type}. Available models: {available}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    creator_func = MODEL_REGISTRY[model_type]

    signature = inspect.signature(creator_func)
    if "random_seed" in signature.parameters:
        return creator_func(random_seed=random_seed, **kwargs)
    else:
        return creator_func(**kwargs)


def get_retraining_kwargs(model_type: ModelType, trained_model: BaseEstimator) -> dict[str, BaseEstimator]:
    """
    Get the retraining kwargs for a model by its type.

    Args:
        model_type: The type of the model.
        trained_model: The trained model to use for retraining.

    Returns:
        dict[str, BaseEstimator]: Dictionary of kwargs for retraining, or empty dict if not supported.
    """
    retraining_keys = {
        ModelType.LINEAR_REGRESSION: None,
        ModelType.XGBOOST_REGRESSOR: "xgb_model",
        ModelType.LIGHTGBM_REGRESSOR: "init_model",
        ModelType.CATBOOST_REGRESSOR: "init_model",
    }

    needs_regressor = {
        ModelType.XGBOOST_REGRESSOR: True,
        ModelType.CATBOOST_REGRESSOR: True,
    }

    key = retraining_keys.get(model_type)
    if not key:
        return {}

    actual_model = trained_model
    if needs_regressor.get(model_type) and isinstance(trained_model, TransformedTargetRegressor):
        actual_model = trained_model.regressor_

    return {key: actual_model}


def wrap_with_transformed_target_regressor(
    model: BaseEstimator, transformer: FunctionTransformer
) -> TransformedTargetRegressor:
    """
    Wrap a model with TransformedTargetRegressor.

    Args:
        model (BaseEstimator): The model to wrap.
        transformer (FunctionTransformer): The transformer to use.

    Returns:
        TransformedTargetRegressor: Wrapped model.
    """
    return TransformedTargetRegressor(regressor=model, transformer=transformer)


def save_model(model: BaseEstimator, model_type: ModelType, models_dir: Path) -> None:
    """
    Save model in the models directory.

    Args:
        model (BaseEstimator): Machine learning model to save.
        model_type (ModelType): Type of the model.
        models_dir (Path): Directory to save the model to.
    """
    model_path = models_dir / f"{model_type}.joblib"
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


def load_model(model_type: ModelType, models_dir: Path) -> BaseEstimator:
    """
    Load model from the models directory.

    Args:
        model_type (ModelType): Type of the model.
        models_dir (Path): Directory to load the model from.

    Returns:
        BaseEstimator: Loaded machine learning model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = models_dir / f"{model_type}.joblib"

    if not model_path.exists():
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    return model


def save_final_model(model: BaseEstimator, model_type: ModelType, ml_task: MLTask) -> None:
    """
    Save the final model to the models directory.

    Args:
        model (BaseEstimator): Machine learning model to save.
        model_type (ModelType): Type of the model.
        ml_task (MLTask): ML task.
    """
    model_dir = PROJECT_DIR / APPDATA_DIRNAME / MODELS_DIRNAME / ml_task / DEFAULT_VERSION
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME
    joblib.dump(model, model_path)

    metadata = {
        "ml_task": ml_task,
        "version": DEFAULT_VERSION,
        "base_version": DEFAULT_VERSION,
        "model": model_type,
        "start_timestamp": -END_TIME,
        "end_timestamp": START_TIME,
    }
    metadata_path = model_dir / METADATA_FILENAME

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Final model and metadata saved to directory {model_dir}")
