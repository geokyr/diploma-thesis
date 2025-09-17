from typing import Optional

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error

from thesis.common.schemas import ErrorPoint, PredictionBatchResponse
from thesis.common.service import PlatformServiceConfig

from .data_loader import TripsDataLoader


class Predictor:
    def __init__(self, model: Optional[BaseEstimator], service_config: PlatformServiceConfig | None = None) -> None:
        self._model = model
        self._cfg = service_config or PlatformServiceConfig()
        self._loader = TripsDataLoader(self._cfg)

    def predict_window(
        self,
        start_timestamp: int,
        end_timestamp: int,
    ) -> PredictionBatchResponse:
        if self._model is None:
            return PredictionBatchResponse(points=[], mae=0.0)

        df = self._loader.load_window(start_timestamp, end_timestamp)
        if df.empty:
            return PredictionBatchResponse(points=[], mae=0.0)

        # Split X, y (target duration) using simple convention
        if "duration" not in df.columns:
            return PredictionBatchResponse(points=[], mae=0.0)

        # Capture timestamps for each sample if available
        ts_col = "time_start" if "time_start" in df.columns else None
        timestamps: list[float] = df[ts_col].astype(float).tolist() if ts_col else []

        X = df.drop(columns=["duration"])  # keep all engineered features
        y = df["duration"].to_numpy()

        # Align columns with model if available
        feature_names: Optional[list[str]] = None
        if hasattr(self._model, "feature_name_"):
            try:
                feature_names = list(getattr(self._model, "feature_name_"))
            except Exception:
                feature_names = None
        try:
            # LightGBM
            booster = getattr(self._model, "booster_", None)
            if booster is not None and hasattr(booster, "feature_name"):
                names = booster.feature_name()
                if names:
                    feature_names = list(names)
        except Exception:
            pass
        try:
            # XGBoost
            booster = getattr(self._model, "get_booster", None)
            if booster is not None:
                b = self._model.get_booster()
                if hasattr(b, "feature_names") and b.feature_names:
                    feature_names = list(b.feature_names)
        except Exception:
            pass

        if feature_names:
            keep = [c for c in feature_names if c in X.columns]
            if keep:
                X = X[keep]

        y_pred = self._model.predict(X)
        abs_errors = (abs(y_pred - y)).tolist()
        mae = float(mean_absolute_error(y, y_pred)) if len(y_pred) > 0 else 0.0

        points: list[ErrorPoint]
        if timestamps and len(timestamps) == len(abs_errors):
            points = [ErrorPoint(timestamp=float(ts), error=float(err)) for ts, err in zip(timestamps, abs_errors)]
        else:
            points = []

        return PredictionBatchResponse(points=points, mae=mae)
