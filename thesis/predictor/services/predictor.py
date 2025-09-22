"""Predictor service for a single model."""

from sklearn.metrics import mean_absolute_error

from thesis.common.config import TIME_START_COLUMN
from thesis.common.schemas import ErrorPoint, PredictionBatchResponse
from thesis.eta.features import split_features_and_target
from thesis.predictor.services.data_loader import ParquetDataLoader
from thesis.predictor.services.model_manager import ModelManager


class Predictor:
    """Predictor service for a single model."""

    def __init__(self, loader: ParquetDataLoader, model_manager: ModelManager) -> None:
        self._loader = loader
        self._model_manager = model_manager

    def predict_window(self, start_timestamp: int, end_timestamp: int) -> PredictionBatchResponse:
        """
        Predict a window of data.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            PredictionBatchResponse: Prediction batch response.
        """
        model = self._model_manager.model
        if model is None:
            return PredictionBatchResponse([], None)

        df = self._loader.load_window(start_timestamp, end_timestamp)
        if df.empty:
            return PredictionBatchResponse([], None)

        X, y = split_features_and_target(df)
        y_pred = model.predict(X)

        timestamps = X[TIME_START_COLUMN].astype(int).tolist()
        abs_errors = (abs(y - y_pred)).tolist()
        mae = mean_absolute_error(y, y_pred)

        if len(timestamps) != len(abs_errors):
            return PredictionBatchResponse([], None)

        points = [ErrorPoint(timestamp, error) for timestamp, error in zip(timestamps, abs_errors)]
        return PredictionBatchResponse(points, mae)

    def close(self) -> None:
        """
        Close the predictor.
        """
        self._model_manager = None
        self._loader = None
