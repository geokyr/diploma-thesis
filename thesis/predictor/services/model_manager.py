from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import joblib
from sklearn.base import BaseEstimator

from thesis.common.service import PlatformServiceConfig


@dataclass(slots=True)
class LoadedModel:
    model: Optional[BaseEstimator]
    version: Optional[str]
    loaded_at: Optional[datetime]


class ModelManager:
    """
    Task-scoped model manager with simple versioning semantics.

    Layout per task:
      /app/models/{task}/
        v1/model.joblib
        v2/model.joblib
        latest -> v2/    (optional symlink)
        latest.txt       (optional text file with version name)
    Legacy fallback: /app/models/LGBMRegressor.joblib
    """

    def __init__(self, service_config: PlatformServiceConfig) -> None:
        self._cfg = service_config
        self._loaded = LoadedModel(model=None, version=None, loaded_at=None)
        self._metadata: Optional[dict[str, Any]] = None

    @property
    def _task_models_dir(self) -> Path:
        return self._cfg.models_dir

    def available_versions(self) -> List[str]:
        if not self._task_models_dir.exists():
            return []
        return sorted(
            [p.name for p in self._task_models_dir.iterdir() if p.is_dir() and p.name.lower().startswith("v")]
        )

    def _resolve_version_dir(self, version: str) -> Optional[Path]:
        # Ignore symlinks; prefer latest.txt when version == "latest"
        if version == "latest":
            latest_txt = self._task_models_dir / "latest.txt"
            if latest_txt.exists():
                try:
                    text = latest_txt.read_text(encoding="utf-8").strip()
                    candidate = self._task_models_dir / text
                    if candidate.is_dir():
                        return candidate
                except Exception:
                    return None
            # Fallback: pick highest vN if latest.txt missing
            versions = self.available_versions()
            if not versions:
                return None
            try:
                versions_sorted = sorted(versions, key=lambda v: int("".join(ch for ch in v if ch.isdigit())))
                return self._task_models_dir / versions_sorted[-1]
            except Exception:
                return self._task_models_dir / versions[-1]

        # explicit version directory
        candidate = self._task_models_dir / version
        return candidate if candidate.is_dir() else None

    def _legacy_model_path(self) -> Path:
        return self._cfg.models_dir / "LGBMRegressor.joblib"

    def load(self, version: str = "latest") -> None:
        # Try versioned path first
        dir_path = self._resolve_version_dir(version)
        if dir_path is not None:
            candidate_names = [
                dir_path / "model.joblib",
                dir_path / "LGBMRegressor.joblib",
                dir_path / "XGBRegressor.joblib",
                dir_path / "CatBoostRegressor.joblib",
            ]
            model_path: Optional[Path] = next((p for p in candidate_names if p.exists()), None)
            if model_path is None:
                # fallback: pick any .joblib in dir
                jobs = list(dir_path.glob("*.joblib"))
                if jobs:
                    model_path = jobs[0]
            if model_path is not None and model_path.exists():
                model = joblib.load(model_path)
                self._loaded = LoadedModel(model=model, version=dir_path.name, loaded_at=datetime.now())
                # try to read metadata.json
                meta_path = dir_path / "metadata.json"
                try:
                    if meta_path.exists():
                        import json

                        self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    else:
                        self._metadata = None
                except Exception:
                    self._metadata = None
                return

        # Legacy fallback
        legacy = self._legacy_model_path()
        if legacy.exists():
            model = joblib.load(legacy)
            self._loaded = LoadedModel(model=model, version="v1", loaded_at=datetime.now())
            self._metadata = None
        else:
            self._loaded = LoadedModel(model=None, version=None, loaded_at=None)
            self._metadata = None

    def load_default(self) -> None:
        self.load("latest")

    @property
    def model(self) -> Optional[BaseEstimator]:
        return self._loaded.model

    @property
    def version(self) -> Optional[str]:
        return self._loaded.version

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded.loaded_at

    @property
    def metadata(self) -> Optional[dict[str, Any]]:
        return self._metadata
