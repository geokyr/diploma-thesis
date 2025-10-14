"""Optuna hyperparameter tuning utilities for ETA prediction models."""

import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from thesis.common.config import DIRECTION, N_TRIALS, RANDOM_SEED_DEFAULT
from thesis.eta.features import FeatureCalibratorETA, split_features_and_target
from thesis.eta.models import ModelType, create_model
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model
from thesis.eta.results import build_cv_results, build_model_results

logger = logging.getLogger(__name__)


class BaseModelTuner(ABC):
    """
    Base class for model hyperparameter tuning using Optuna.

    Attributes:
        model_type (ModelType): Type of model to tune.
        n_trials (int): Number of optimization trials.
        random_seed (int): Random seed for reproducibility.
        direction (str): Optimization direction ("minimize" or "maximize").
        trips_train_raw (pd.DataFrame | None): Training trips before feature engineering.
        skf (StratifiedKFold | None): StratifiedKFold object for cross-validation.
        stratify_key (pd.Series | None): Key for stratified cross-validation.
        study_name (str | None): Name for the study.
        fit_kwargs (dict): Additional fitting arguments.
    """

    def __init__(
        self,
        model_type: ModelType,
        n_trials: int = N_TRIALS,
        random_seed: int = RANDOM_SEED_DEFAULT,
        direction: str = DIRECTION,
    ) -> None:
        self.model_type = model_type
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.direction = direction

        self.trips_train_raw = None
        self.skf = None
        self.stratify_key = None
        self.study_name = None
        self.fit_kwargs = {}

    def setup_data(
        self,
        trips_train_raw: pd.DataFrame,
        skf: StratifiedKFold,
        stratify_key: pd.Series,
        study_name: str,
        **fit_kwargs,
    ) -> None:
        """
        Setup training data and cross-validation strategy.

        Args:
            trips_train_raw (pd.DataFrame): Training trips before feature engineering.
            skf (StratifiedKFold): StratifiedKFold object for cross-validation.
            stratify_key (pd.Series): Key for stratified cross-validation.
            study_name (str): Name for the study.
            **fit_kwargs: Additional fitting arguments.
        """
        self.trips_train_raw = trips_train_raw
        self.skf = skf
        self.stratify_key = stratify_key
        self.study_name = study_name
        self.fit_kwargs = fit_kwargs

    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """
        Suggest hyperparameters for the current trial.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            dict[str, int | float]: Dictionary of hyperparameters.
        """
        pass

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Mean cross-validation MAE score.
        """
        hyperparams = self.suggest_hyperparameters(trial)

        logger.info(f"Starting trial {trial.number} with parameters: {hyperparams}")

        model = create_model(self.model_type, random_seed=self.random_seed, **hyperparams)

        per_fold_results = []
        fold_maes = []

        for train_index, validation_index in self.skf.split(self.trips_train_raw, self.stratify_key):
            trips_train_fold_raw = self.trips_train_raw.iloc[train_index]
            trips_validation_fold_raw = self.trips_train_raw.iloc[validation_index]

            calibrator = FeatureCalibratorETA.from_train_trips(trips_train_fold_raw)

            trips_train_fold = calibrator.transform(trips_train_fold_raw)
            trips_validation_fold = calibrator.transform(trips_validation_fold_raw)

            X_train_fold, y_train_fold = split_features_and_target(trips_train_fold)
            X_validation_fold, y_validation_fold = split_features_and_target(trips_validation_fold)

            training_results = train_model(model, self.model_type, X_train_fold, y_train_fold, **self.fit_kwargs)
            predictions, prediction_results = make_predictions(model, self.model_type, X_validation_fold)
            evaluation_results = evaluate_predictions(y_validation_fold, predictions, self.model_type)

            fold_results = build_model_results(training_results, prediction_results, evaluation_results)
            per_fold_results.append(fold_results)

            fold_mae = evaluation_results.mae
            fold_maes.append(fold_mae)

        cv_results = build_cv_results(per_fold_results)

        trial.set_user_attr("mae", cv_results.mae)
        trial.set_user_attr("mape", cv_results.mape)
        trial.set_user_attr("training_time", cv_results.training_time)
        trial.set_user_attr("prediction_time", cv_results.prediction_time)
        trial.set_user_attr("evaluation_time", cv_results.evaluation_time)

        logger.info(f"Trial {trial.number} finished with value: {cv_results.mae} and parameters: {hyperparams}.")
        logger.info(
            f"Training time: {cv_results.training_time:.3f}s, Prediction time: {cv_results.prediction_time:.3f}s, Evaluation time: {cv_results.evaluation_time:.3f}s, MAE: {cv_results.mae:.2f}s, MAPE: {cv_results.mape * 100:.2f}%"
        )

        return cv_results.mae

    def optimize(self) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Returns:
            optuna.Study: Completed Optuna study.
        """
        if self.trips_train_raw is None:
            raise ValueError("Data not setup. Call setup_data() first.")

        logger.info(f"Starting {self.model_type} hyperparameter tuning with {self.n_trials} trials")

        study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
        )
        study.set_user_attr("model", self.model_type)
        study.optimize(self.objective, n_trials=self.n_trials)

        total_trials = len(study.trials)
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        logger.info("Hyperparameter tuning completed")
        logger.info(f"  Total trials: {total_trials}")
        logger.info(f"  Completed trials: {completed_trials}")
        logger.info(f"  Best trial: #{study.best_trial.number}")
        logger.info(f"  Best MAE: {study.best_value:.2f}s")
        logger.info(f"  Best parameters: {study.best_params}")

        if study.best_trial.user_attrs:
            attrs = study.best_trial.user_attrs
            logger.info("Best trial metrics:")
            logger.info(f"  MAE: {attrs.get('mae', 'N/A'):.2f}s")
            logger.info(f"  MAPE: {attrs.get('mape', 'N/A') * 100:.2f}%")
            logger.info(f"  Training time: {attrs.get('training_time', 'N/A'):.3f}s")
            logger.info(f"  Prediction time: {attrs.get('prediction_time', 'N/A'):.3f}s")
            logger.info(f"  Evaluation time: {attrs.get('evaluation_time', 'N/A'):.3f}s")

        return study


class XGBoostTuner(BaseModelTuner):
    """XGBoost class for model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.XGBOOST_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest XGBoost hyperparameters for the current trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }


class XGBoostTunerFocused(BaseModelTuner):
    """XGBoost class for focused model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.XGBOOST_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest XGBoost hyperparameters for the current trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 900, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.7),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 0.85),
            "min_child_weight": trial.suggest_int("min_child_weight", 8, 12),
            "gamma": trial.suggest_float("gamma", 1e-8, 0.6, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }


class LightGBMTuner(BaseModelTuner):
    """LightGBM class for model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.LIGHTGBM_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest LightGBM hyperparameters for the current trial."""
        max_depth = trial.suggest_int("max_depth", 3, 12)
        max_leaves = max(20, min(300, 2**max_depth))

        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 20, max_leaves),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }


class LightGBMTunerFocused(BaseModelTuner):
    """LightGBM class for focused model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.LIGHTGBM_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest LightGBM hyperparameters for the current trial."""
        max_depth = trial.suggest_int("max_depth", 10, 14)
        max_leaves = max(100, min(170, 2**max_depth))

        return {
            "n_estimators": trial.suggest_int("n_estimators", 900, 1200, step=50),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 100, max_leaves),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.9, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 35, 60),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 0.5, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 0.1, log=True),
        }


class CatBoostTuner(BaseModelTuner):
    """CatBoost class for model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.CATBOOST_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest CatBoost hyperparameters for the current trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 255]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        }


class CatBoostTunerFocused(BaseModelTuner):
    """CatBoost class for focused model hyperparameter tuning using Optuna."""

    def __init__(self, **kwargs):
        super().__init__(ModelType.CATBOOST_REGRESSOR, **kwargs)

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict[str, int | float]:
        """Suggest CatBoost hyperparameters for the current trial."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 900, 1200, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.9, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1.0),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 1.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 3.0, 5.5),
            "border_count": trial.suggest_categorical("border_count", [255]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 45),
        }


def run_hyperparameter_tuning(
    tuner: BaseModelTuner,
    trips_train_raw: pd.DataFrame,
    skf: StratifiedKFold,
    stratify_key: pd.Series,
    study_name: str,
) -> optuna.Study:
    """
    Run hyperparameter tuning with the provided tuner.

    Args:
        tuner (BaseModelTuner): The tuner instance to use.
        trips_train_raw (pd.DataFrame): Training trips before feature engineering.
        skf (StratifiedKFold): StratifiedKFold object for cross-validation.
        stratify_key (pd.Series): Key for stratified cross-validation.
        study_name (str): Name for the study.

    Returns:
        optuna.Study: Completed Optuna study.
    """
    tuner.setup_data(trips_train_raw, skf, stratify_key, study_name)
    study = tuner.optimize()
    log_parameter_importance(study)

    return study


def log_parameter_importance(study: optuna.Study) -> None:
    """
    Log parameter importance from the study.

    Args:
        study (optuna.Study): Completed Optuna study.
    """
    importance = optuna.importance.get_param_importances(study)
    logger.info("Parameter importance:")
    for param, imp in importance.items():
        logger.info(f"  {param}: {imp:.4f}")
