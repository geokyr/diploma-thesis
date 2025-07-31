import inspect
import logging
from enum import StrEnum
from typing import Callable

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer
from xgboost import XGBRegressor

from thesis.common.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class ModelType(StrEnum):
    """
    Model Types.

    Attributes:
        LINEAR_REGRESSION: Linear Regression.
        XGBOOST_REGRESSOR: XGBoost Regressor.
        LIGHTGBM_REGRESSOR: LightGBM Regressor.
        CATBOOST_REGRESSOR: CatBoost Regressor.
    """

    LINEAR_REGRESSION = "LinearRegression"
    XGBOOST_REGRESSOR = "XGBRegressor"
    LIGHTGBM_REGRESSOR = "LGBMRegressor"
    CATBOOST_REGRESSOR = "CatBoostRegressor"


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
def create_xgboost_regressor_model(random_seed: int = RANDOM_SEED, **kwargs) -> XGBRegressor:
    """
    Create an XGBoost Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        XGBRegressor: Configured XGBoost Regressor model.
    """
    params = {
        "objective": "reg:absoluteerror",
        "random_state": random_seed,
        **kwargs,
    }

    return XGBRegressor(**params)


@register_model(ModelType.LIGHTGBM_REGRESSOR)
def create_lightgbm_regressor_model(random_seed: int = RANDOM_SEED, **kwargs) -> LGBMRegressor:
    """
    Create a LightGBM Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LGBMRegressor: Configured LightGBM Regressor model.
    """
    params = {
        "objective": "regression_l1",
        "random_state": random_seed,
        "verbose": 0,
        **kwargs,
    }

    return LGBMRegressor(**params)


@register_model(ModelType.CATBOOST_REGRESSOR)
def create_catboost_regressor_model(random_seed: int = RANDOM_SEED, **kwargs) -> CatBoostRegressor:
    """
    Create a CatBoost Regressor model with a default configuration.

    Args:
        random_seed (int): The random seed to use for the random number generator.
        **kwargs: Additional parameters to pass to the model.

    Returns:
        CatBoostRegressor: Configured CatBoost Regressor model.
    """
    params = {
        "loss_function": "MAE",
        "random_state": random_seed,
        "verbose": 0,
        "allow_writing_files": False,
        **kwargs,
    }

    return CatBoostRegressor(**params)


def create_model(model_type: ModelType, random_seed: int = RANDOM_SEED, **kwargs) -> BaseEstimator:
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
        raise ValueError(f"Unknown model: {model_type}. Available models: {available}")

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

    key = retraining_keys.get(model_type)
    return {key: trained_model} if key else {}


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
