"""Model manager for a predictor with simple versioning."""

from pathlib import Path
from threading import RLock

import joblib
from sklearn.base import BaseEstimator

from thesis.common.config import DEFAULT_VERSION, MODEL_FILENAME


# TODO: add async
class ModelManager:
    """
    Model manager for a predictor with simple versioning.

    Attributes:
        model (BaseEstimator | None): The loaded model.
        version (str | None): The version of the loaded model.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir: Path = models_dir
        self._lock: RLock = RLock()
        self.version: str | None = None
        self.model: BaseEstimator | None = None

        self.load()

    def load(self, version: str = DEFAULT_VERSION) -> bool:
        """
        Load a model for a given version.

        Args:
            version (str): The version of the model to load.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        with self._lock:
            if version == self.version:
                return True

            model_path = self._models_dir / version / MODEL_FILENAME
            if not model_path.exists():
                return False

            model = joblib.load(model_path)
            version = model_path.parent.name

            self.model = model
            self.version = version
            return True

    def clear(self) -> None:
        """Clear the model manager."""
        with self._lock:
            self.version = None
            self.model = None
