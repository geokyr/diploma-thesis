from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from thesis.eta.config import RANDOM_STATE


def create_lr_model(**kwargs) -> LinearRegression:
    """
    Create a Linear Regression model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LinearRegression: Configured Linear Regression model.
    """
    return LinearRegression(**kwargs)


def create_mlp_model(**kwargs) -> MLPRegressor:
    """
    Create a Multi-Layer Perceptron model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        MLPRegressor: Configured MLP model.
    """
    return MLPRegressor(random_state=RANDOM_STATE, **kwargs)


def create_xgboost_model(**kwargs) -> XGBRegressor:
    """
    Create an XGBoost model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        XGBRegressor: Configured XGBoost model.
    """
    return XGBRegressor(random_state=RANDOM_STATE, **kwargs)


def create_lightgbm_model(**kwargs) -> LGBMRegressor:
    """
    Create a LightGBM model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        LGBMRegressor: Configured LightGBM model.
    """
    return LGBMRegressor(random_state=RANDOM_STATE, verbose=0, **kwargs)


def create_catboost_model(**kwargs) -> CatBoostRegressor:
    """
    Create a CatBoost model with a default configuration.

    Args:
        **kwargs: Additional parameters to pass to the model.

    Returns:
        CatBoostRegressor: Configured CatBoost model.
    """
    return CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False, **kwargs)


def get_baseline_models() -> dict[str, BaseEstimator]:
    """
    Get a dictionary of baseline models.

    Returns:
        dict[str, BaseEstimator]: Dictionary of baseline models.
    """
    return {
        "lr": create_lr_model(),
        "mlp": create_mlp_model(),
        "xgboost": create_xgboost_model(),
        "lightgbm": create_lightgbm_model(),
        "catboost": create_catboost_model(),
    }
