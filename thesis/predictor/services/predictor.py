"""Predictor service for a single model."""

import pandas as pd
from click import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from thesis.common.config import (
    TIME_START_COLUMN,
)
from thesis.common.enums import MLTask
from thesis.common.schemas import ErrorPoint, PredictionBatchResponse, PredictionSingleResponse
from thesis.eta.features import FeatureCalibrator, split_features_and_target
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.sumo_service import SumoService


class Predictor:
    """Predictor service for a single model."""

    def __init__(
        self,
        ml_task: MLTask,
        misc_dir: Path,
        data_loader: DataLoader,
        model_manager: ModelManager,
        sumo_service: SumoService,
    ) -> None:
        self._ml_task: MLTask = ml_task
        self._feature_calibrator: FeatureCalibrator = FeatureCalibrator.load(misc_dir)
        self._data_loader: DataLoader = data_loader
        self._model_manager: ModelManager = model_manager
        self._sumo_service: SumoService = sumo_service

    def predict_window(self, start_timestamp: int, end_timestamp: int) -> PredictionBatchResponse:
        """
        Predict a window of data.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            PredictionBatchResponse: Prediction batch response.
        """
        model: BaseEstimator = self._model_manager.model

        df = self._data_loader.load_window(start_timestamp, end_timestamp)
        if df.empty:
            return PredictionBatchResponse(error_points=[], mae=None)

        X, y = split_features_and_target(df)
        y_pred = model.predict(X)

        timestamps = X[TIME_START_COLUMN].astype(int).tolist()
        abs_errors = (abs(y - y_pred)).tolist()
        mae = mean_absolute_error(y, y_pred)

        if len(timestamps) != len(abs_errors):
            return PredictionBatchResponse(error_points=[], mae=None)

        error_points = [
            ErrorPoint(timestamp=timestamp, error=error) for timestamp, error in zip(timestamps, abs_errors)
        ]
        return PredictionBatchResponse(error_points=error_points, mae=mae)

    def predict_single(
        self,
        source_latitude: float,
        source_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        start_timestamp: int,
    ) -> PredictionSingleResponse:
        """
        Predict a single trip.

        Args:
            source_latitude (float): Source latitude.
            source_longitude (float): Source longitude.
            destination_latitude (float): Destination latitude.
            destination_longitude (float): Destination longitude.
            start_time (int): Trip start time.

        Returns:
            PredictionSingleResponse: Single trip prediction response.
        """
        model: BaseEstimator = self._model_manager.model
        trip_data = {}

        if self._ml_task == MLTask.ETA:
            source_x, source_y = self._sumo_service.lonlat_to_xy(source_longitude, source_latitude)
            destination, destination_y = self._sumo_service.lonlat_to_xy(destination_longitude, destination_latitude)
            distance = self._sumo_service.calculate_trip_distance(source_x, source_y, destination, destination_y)

            # TODO: add columns to config
            trip_data = {
                "source_x": [source_x],
                "source_y": [source_y],
                "destination_x": [destination],
                "destination_y": [destination_y],
                TIME_START_COLUMN: [start_timestamp],
                "distance": [distance],
            }

        X = self._feature_calibrator.transform(pd.DataFrame(trip_data))
        prediction = model.predict(X)[0]

        return PredictionSingleResponse(prediction=float(prediction))

    def clear(self) -> None:
        """Clear the predictor."""
        self._feature_calibrator = None
