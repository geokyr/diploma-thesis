"""Predictor service for a single model."""

import numpy as np
import pandas as pd
from click import Path
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from thesis.common.config import (
    DESTINATION_X_COLUMN_ETA,
    DESTINATION_Y_COLUMN_ETA,
    DISTANCE_COLUMN_ETA,
    SOURCE_X_COLUMN_ETA,
    SOURCE_Y_COLUMN_ETA,
    TARGET_COLUMN_ETA,
    TARGET_COLUMN_FUEL,
    TIME_START_COLUMN_ETA,
)
from thesis.common.enums import MLTask
from thesis.common.schemas import (
    ErrorPoint,
    PredictionBatchResponse,
    PredictionSingleResponse,
    RoutePreviewResponse,
)
from thesis.eta.features import FeatureCalibratorETA, split_features_and_target
from thesis.fuel.features import FeatureCalibratorFuel
from thesis.predictor.services.data_loader import _TIME_START_COLUMN_MAP, DataLoader
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.sumo_service import SumoService

# TODO: add stops
_FEATURE_CALIBRATOR_MAP: dict[MLTask, FeatureCalibratorETA | FeatureCalibratorFuel | None] = {
    MLTask.ETA: FeatureCalibratorETA,
    MLTask.FUEL: FeatureCalibratorFuel,
    MLTask.STOPS: None,
}

# TODO: add stops
_TARGET_COLUMN_MAP: dict[MLTask, str] = {
    MLTask.ETA: TARGET_COLUMN_ETA,
    MLTask.FUEL: TARGET_COLUMN_FUEL,
    MLTask.STOPS: "",
}


class Predictor:
    """Predictor service for a single model."""

    def __init__(
        self,
        ml_task: MLTask,
        misc_dir: Path,
        model_manager: ModelManager,
        data_loader: DataLoader,
        sumo_service: SumoService,
    ) -> None:
        self._ml_task: MLTask = ml_task
        self._feature_calibrator: FeatureCalibratorETA | FeatureCalibratorFuel | None = _FEATURE_CALIBRATOR_MAP[
            ml_task
        ].load(misc_dir)

        self._model_manager: ModelManager = model_manager
        self._data_loader: DataLoader = data_loader
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

        X, y = split_features_and_target(df, target_columns=[_TARGET_COLUMN_MAP[self._ml_task]])
        y_pred = model.predict(X)

        time_column = _TIME_START_COLUMN_MAP[self._ml_task]
        timestamps = X[time_column].astype(int).tolist()

        if self._ml_task == MLTask.FUEL:
            y = np.expm1(y)
            y_pred = np.expm1(y_pred)

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
        start_timestamp: int,
        source_x: float,
        source_y: float,
        destination_x: float,
        destination_y: float,
        distance: float,
        edges: list[str],
        minimum_x: float,
        maximum_x: float,
        minimum_y: float,
        maximum_y: float,
    ) -> PredictionSingleResponse:
        """
        Predict a single trip.

        Args:
            start_timestamp (int): Trip start time.
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            destination_x (float): Destination x coordinate.
            destination_y (float): Destination y coordinate.
            distance (float): Trip distance in meters.
            edges (list[str]): List of edge IDs along the trip.
            minimum_x (float): Minimum x coordinate along the route.
            maximum_x (float): Maximum x coordinate along the route.
            minimum_y (float): Minimum y coordinate along the route.
            maximum_y (float): Maximum y coordinate along the route.

        Returns:
            PredictionSingleResponse: Single trip prediction response.
        """
        model: BaseEstimator = self._model_manager.model
        trip_data = {}

        if self._ml_task == MLTask.ETA:
            trip_data = {
                SOURCE_X_COLUMN_ETA: [source_x],
                SOURCE_Y_COLUMN_ETA: [source_y],
                DESTINATION_X_COLUMN_ETA: [destination_x],
                DESTINATION_Y_COLUMN_ETA: [destination_y],
                TIME_START_COLUMN_ETA: [start_timestamp],
                DISTANCE_COLUMN_ETA: [distance],
            }
        elif self._ml_task == MLTask.FUEL:
            trip_data = {
                "timestep_time_min": [start_timestamp],
                "vehicle_x_first": [source_x],
                "vehicle_y_first": [source_y],
                "vehicle_x_last": [destination_x],
                "vehicle_y_last": [destination_y],
                "trip_actual_distance": [distance],
                "minimum_x": [minimum_x],
                "maximum_x": [maximum_x],
                "minimum_y": [minimum_y],
                "maximum_y": [maximum_y],
            }
        elif self._ml_task == MLTask.STOPS:
            # TODO: add stops
            pass

        X = self._feature_calibrator.transform(pd.DataFrame(trip_data))
        prediction = model.predict(X)[0]

        if self._ml_task == MLTask.FUEL:
            prediction = np.expm1(prediction)

        return PredictionSingleResponse(prediction=float(prediction))

    def get_route(
        self,
        source_latitude: float,
        source_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> RoutePreviewResponse:
        """
        Get route polyline and all computed features for the given source and destination.

        Args:
            source_latitude (float): Source latitude.
            source_longitude (float): Source longitude.
            destination_latitude (float): Destination latitude.
            destination_longitude (float): Destination longitude.

        Returns:
            RoutePreviewResponse: Route polyline and computed features.
        """
        features = self._sumo_service.compute_trip_features(
            source_latitude, source_longitude, destination_latitude, destination_longitude
        )

        return RoutePreviewResponse(
            source_x=features["source_x"],
            source_y=features["source_y"],
            destination_x=features["destination_x"],
            destination_y=features["destination_y"],
            distance=features["distance"],
            edges=features["edges"],
            route=features["route"],
            minimum_x=features["minimum_x"],
            maximum_x=features["maximum_x"],
            minimum_y=features["minimum_y"],
            maximum_y=features["maximum_y"],
        )

    def clear(self) -> None:
        """Clear the predictor."""
        self._feature_calibrator = None
