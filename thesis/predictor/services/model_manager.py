"""Model manager for a predictor with simple versioning."""

import json
from pathlib import Path
from threading import RLock

import joblib
from sklearn.base import BaseEstimator

from thesis.common.config import DEFAULT_VERSION, METADATA_FILENAME, MODEL_FILENAME
from thesis.eta.models import ModelType


# TODO: add async
class ModelManager:
    """Model manager for a predictor with simple versioning.

    Attributes:
        models_dir (Path): Directory to save and load models from.
        model (BaseEstimator | None): Loaded model.
        version (str | None): Version of the loaded model.
        metadata (dict[str, str | int] | None): Loaded model metadata for the current version.
    """

    def __init__(self, models_dir: Path, version: str = DEFAULT_VERSION) -> None:
        self._lock: RLock = RLock()
        self.models_dir: Path = models_dir
        self.model: BaseEstimator | None = None
        self.version: str | None = None
        self.metadata: dict[str, str | int] | None = None

        self.load(version)

    def load(self, version: str = DEFAULT_VERSION) -> bool:
        """Load a model for a given version.

        Args:
            version (str): Version of the model to load.

        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        with self._lock:
            if version == self.version:
                return True

            model_path = self.models_dir / version / MODEL_FILENAME
            if not model_path.exists():
                return False

            model = joblib.load(model_path)
            version = model_path.parent.name
            metadata = self._read_metadata(version)

            self.model = model
            self.version = version
            self.metadata = metadata
            return True

    def save(self, model: BaseEstimator, version: str, metadata: dict[str, str | int]) -> None:
        """Save a model under a directory for a given version.

        Args:
            model (BaseEstimator): Trained model.
            version (str): Version of the model.
            metadata (dict[str, str | int]): Metadata of the model.
        """
        with self._lock:
            model_dir = self.models_dir / version
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / MODEL_FILENAME
            joblib.dump(model, model_path)

            self._write_metadata(model_dir, metadata)

    def get_next_version(self) -> str:
        """Compute the next version by scanning existing version directories.

        Returns:
            str: Next version string.
        """
        with self._lock:
            max_num = 0

            for entry in self.models_dir.iterdir():
                if not entry.is_dir():
                    continue

                name = entry.name
                if not name.startswith("v"):
                    continue

                suffix = name[1:]
                if not suffix.isdigit():
                    continue

                n = int(suffix)
                if n > max_num:
                    max_num = n

            return f"v{max_num + 1}"

    def _read_metadata(self, version: str = DEFAULT_VERSION) -> dict[str, str | int] | None:
        """Read metadata for a given version.

        Args:
            version (str): Version to read metadata for.

        Returns:
            dict[str, str | int] | None: Loaded metadata.
        """
        with self._lock:
            metadata_path = self.models_dir / version / METADATA_FILENAME
            if not metadata_path.exists():
                return None

            with open(metadata_path, "r") as f:
                return json.load(f)

    def _write_metadata(self, model_dir: Path, metadata: dict[str, str | int]) -> None:
        """Write metadata under the model directory.

        Args:
            model_dir (Path): Directory of the model.
            metadata (dict[str, str | int]): Metadata of the model.
        """
        with self._lock:
            metadata_path = model_dir / METADATA_FILENAME
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

    def get_model_type(self) -> ModelType:
        """Return ModelType parsed from current metadata.

        Returns:
            ModelType: ModelType parsed from current metadata.
        """
        with self._lock:
            metadata = self.metadata
            raw = metadata["model"]
            return ModelType(raw)

    def clear(self) -> None:
        """Clear the model manager."""
        with self._lock:
            self.version = None
            self.model = None
            self.metadata = None
