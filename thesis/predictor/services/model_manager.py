"""Model manager for a predictor with simple versioning."""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from thesis.common.config import DEFAULT_VERSION, MODEL_FILENAME


class ModelManager:
    """
    Model manager for a predictor with simple versioning.

    Attributes:
        model (BaseEstimator | None): The loaded model.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir: Path | None = models_dir
        self._version: str | None = None
        self.model: BaseEstimator | None = None

        self.load()

    def load(self, version: str = DEFAULT_VERSION) -> None:
        """
        Load a model for a given version.

        Args:
            version (str): The version of the model to load.
        """
        if version == self._version:
            return

        model_path = self._models_dir / version / MODEL_FILENAME
        if not model_path.exists():
            return

        self.model = joblib.load(model_path)
        self._version = model_path.parent.name

    def close(self) -> None:
        """
        Close the model manager.
        """
        self._models_dir = None
        self._version = None
        self.model = None
