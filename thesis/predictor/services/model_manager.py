"""Model manager for a predictor with simple versioning."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from thesis.common.config import LATEST_FILENAME, LATEST_VERSION, METADATA_FILENAME, MODEL_FILENAME
from thesis.common.enums import MLTask
from thesis.eta.models import ModelType


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """
    Metadata for a model.

    Attributes:
        ml_task (MLTask): The task the model is for.
        version (str): The version of the model.
        created_at (str): The timestamp the model was created at.
        model (ModelType): The model type.
        start_timestamp (int): The training data start timestamp used to train the model.
        end_timestamp (int): The training data end timestamp used to train the model.
    """

    ml_task: MLTask
    version: str
    created_at: str
    model: ModelType
    start_timestamp: int
    end_timestamp: int


class ModelManager:
    """
    Model manager for a predictor with simple versioning.

    Attributes:
        model (BaseEstimator | None): The loaded model.
    """

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = models_dir
        self._metadata: ModelMetadata | None = None
        self._version: str | None = None

        self.model: BaseEstimator | None = None

        self.load()

    def _read_latest_version(self) -> str | None:
        """
        Read the latest version from the latest.txt file.

        Returns:
            str | None: The latest version.
        """
        latest_txt = self._models_dir / LATEST_FILENAME
        if not latest_txt.exists():
            return None

        try:
            return latest_txt.read_text(encoding="utf-8").strip()
        except Exception:
            return None

    def _resolve_model_path(self, version: str = LATEST_VERSION) -> Path | None:
        """
        Resolve the model path for a given version.

        Args:
            version (str): The version of the model to resolve.

        Returns:
            Path | None: The resolved model path.
        """
        if version == LATEST_VERSION:
            latest_version = self._read_latest_version()
            if latest_version is None:
                return None

            version = latest_version

        model_path = self._models_dir / version / MODEL_FILENAME
        return model_path if model_path.exists() else None

    def load(self, version: str = LATEST_VERSION) -> None:
        """
        Load a model for a given version.

        Args:
            version (str): The version of the model to load.
        """
        if version == self._version:
            return

        if version == LATEST_VERSION and self._version == self._read_latest_version():
            return

        model_path = self._resolve_model_path(version)
        if model_path is None:
            return

        self.model = joblib.load(model_path)
        self._version = model_path.parent.name

        metadata_path = model_path.with_name(METADATA_FILENAME)

        try:
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._metadata = ModelMetadata(**metadata)
        except Exception:
            self._metadata = None
