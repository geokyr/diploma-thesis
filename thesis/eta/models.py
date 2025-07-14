from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FunctionTransformer
from xgboost import XGBRegressor

from thesis.common.config import RANDOM_SEED
from thesis.eta.config import CATBOOST, LIGHTGBM, LR, XGBOOST


def create_lr_model(**kwargs) -> LinearRegression:
    """
    Create a Linear Regression model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LinearRegression: Configured Linear Regression model.
    """
    return LinearRegression(**kwargs)


def create_xgboost_model(**kwargs) -> XGBRegressor:
    """
    Create an XGBoost model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        XGBRegressor: Configured XGBoost model.
    """
    params = {
        "random_state": RANDOM_SEED,
        **kwargs,
    }

    return XGBRegressor(**params)


def create_lightgbm_model(**kwargs) -> LGBMRegressor:
    """
    Create a LightGBM model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LGBMRegressor: Configured LightGBM model.
    """
    params = {
        "random_state": RANDOM_SEED,
        "verbose": 0,
        **kwargs,
    }

    return LGBMRegressor(**params)


def create_catboost_model(**kwargs) -> CatBoostRegressor:
    """
    Create a CatBoost model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        CatBoostRegressor: Configured CatBoost model.
    """
    params = {
        "random_state": RANDOM_SEED,
        "verbose": 0,
        "allow_writing_files": False,
        **kwargs,
    }

    return CatBoostRegressor(**params)


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


def get_baseline_models() -> dict[str, BaseEstimator]:
    """
    Get a dictionary of baseline models.

    Returns:
        dict[str, BaseEstimator]: Dictionary of baseline models.
    """
    return {
        LR: create_lr_model(),
        XGBOOST: create_xgboost_model(),
        LIGHTGBM: create_lightgbm_model(),
        CATBOOST: create_catboost_model(),
    }
