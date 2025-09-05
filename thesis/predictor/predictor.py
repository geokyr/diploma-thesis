"""
Model loading and prediction service.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from thesis.eta.features import optimize_features_for_model
from thesis.eta.models import ModelType
from thesis.predictor.preprocessing import (
    prepare_features_for_prediction,
    preprocess_trip_batch,
    validate_trip_data,
)

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading, retraining, and prediction for a specific task."""

    def __init__(self, task_type: str, models_path: Path):
        self.task_type = task_type  # "eta", "fuel", "stops"
        self.models_path = models_path / task_type

        self.current_model = None
        self.current_version = "stable"
        self.model_metadata = {}

        # Ensure model directories exist
        (self.models_path / "stable").mkdir(parents=True, exist_ok=True)
        (self.models_path / "retrained").mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {task_type} predictor with models path: {self.models_path}")

    async def load_model(self, version: str = "stable") -> bool:
        """
        Load model from filesystem.

        Args:
            version: "stable" or "retrained"

        Returns:
            True if model loaded successfully
        """
        try:
            model_path = self.models_path / version / "model.joblib"
            metadata_path = self.models_path / version / "metadata.json"

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Load model
            logger.info(f"Loading {self.task_type} model from {model_path}")
            self.current_model = joblib.load(model_path)
            self.current_version = version

            # Load metadata if available
            if metadata_path.exists():
                import json

                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                logger.debug(f"Loaded model metadata: {self.model_metadata}")

            logger.info(f"Successfully loaded {self.task_type} model version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load {self.task_type} model: {e}")
            return False

    async def predict_batch(self, trip_data: List[dict]) -> List[float]:
        """
        Predict on batch of trips.

        Args:
            trip_data: List of trip dictionaries

        Returns:
            List of predictions
        """
        try:
            if self.current_model is None:
                raise ValueError(f"No model loaded for {self.task_type}")

            # Validate input data
            if not validate_trip_data(trip_data):
                raise ValueError("Invalid trip data")

            # Preprocess trips to features
            df_features = preprocess_trip_batch(trip_data)

            if df_features.empty:
                return []

            # Prepare for prediction (remove target columns)
            df_prediction = prepare_features_for_prediction(df_features)

            # Optimize features for model type (if we can detect the model type)
            try:
                model_type = self._detect_model_type()
                if model_type:
                    df_optimized, _ = optimize_features_for_model(df_prediction, model_type)
                else:
                    df_optimized = df_prediction
            except Exception as e:
                logger.warning(f"Feature optimization failed, using raw features: {e}")
                df_optimized = df_prediction

            # Make predictions
            predictions = self.current_model.predict(df_optimized)

            # Convert to list and handle any numpy types
            if isinstance(predictions, np.ndarray):
                predictions_list = predictions.tolist()
            else:
                predictions_list = list(predictions)

            logger.debug(f"Generated {len(predictions_list)} predictions for {self.task_type}")
            return predictions_list

        except Exception as e:
            logger.error(f"Prediction failed for {self.task_type}: {e}")
            raise

    def _detect_model_type(self) -> Optional[ModelType]:
        """Try to detect the model type from the loaded model."""
        try:
            if self.current_model is None:
                return None

            # Check model class name
            model_class = self.current_model.__class__.__name__

            # Handle sklearn pipeline wrapper
            if hasattr(self.current_model, "regressor_"):
                model_class = self.current_model.regressor_.__class__.__name__
            elif hasattr(self.current_model, "steps"):
                # sklearn pipeline
                model_class = self.current_model.steps[-1][1].__class__.__name__

            # Map to ModelType enum
            type_mapping = {
                "LGBMRegressor": ModelType.LIGHTGBM_REGRESSOR,
                "XGBRegressor": ModelType.XGBOOST_REGRESSOR,
                "CatBoostRegressor": ModelType.CATBOOST_REGRESSOR,
            }

            return type_mapping.get(model_class)

        except Exception as e:
            logger.warning(f"Could not detect model type: {e}")
            return None

    def get_status(self) -> Dict:
        """Get current predictor status."""
        return {
            "task_type": self.task_type,
            "model_loaded": self.current_model is not None,
            "current_version": self.current_version,
            "model_metadata": self.model_metadata,
        }

    async def retrain_model(self, retrain_data: Dict) -> bool:
        """
        Placeholder for retraining functionality.
        This will be implemented in Phase 3.
        """
        logger.info(f"Retrain requested for {self.task_type} (not implemented yet)")
        # TODO: Implement retraining logic in Phase 3
        return True
