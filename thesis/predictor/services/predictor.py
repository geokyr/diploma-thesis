"""Predictor service for a single model."""

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from thesis.common.config import TIME_START_COLUMN
from thesis.common.schemas import ErrorPoint, PredictionBatchResponse
from thesis.eta.features import split_features_and_target
from thesis.predictor.services.data_loader import ParquetDataLoader


class Predictor:
    """Predictor service for a single model."""

    def __init__(self, model: BaseEstimator | None, loader: ParquetDataLoader) -> None:
        self._model = model
        self._loader = loader

    def predict_window(self, start_timestamp: int, end_timestamp: int) -> PredictionBatchResponse:
        """
        Predict a window of data.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            PredictionBatchResponse: Prediction batch response.
        """
        if self._model is None:
            return PredictionBatchResponse(points=[], mae=None)

        df = self._loader.load_window(start_timestamp, end_timestamp)
        if df.empty:
            return PredictionBatchResponse(points=[], mae=None)

        X, y = split_features_and_target(df)
        y_pred = self._model.predict(X)

        timestamps = X[TIME_START_COLUMN].astype(int).tolist()
        abs_errors = (abs(y - y_pred)).tolist()
        mae = float(mean_absolute_error(y, y_pred))

        if len(timestamps) != len(abs_errors):
            return PredictionBatchResponse(points=[], mae=None)

        points = [ErrorPoint(timestamp=timestamp, error=error) for timestamp, error in zip(timestamps, abs_errors)]
        return PredictionBatchResponse(points=points, mae=mae)
