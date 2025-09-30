"""Retraining service for predictor."""

import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

from sklearn.base import clone

from thesis.common.enums import RetrainStatus
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_retraining_kwargs
from thesis.eta.pipeline import train_model
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager


def _retrain_job(
    data_dir: Path, models_dir: Path, base_version: str, version: str, start_timestamp: int, end_timestamp: int
) -> None:
    """Retrain job.

    Args:
        data_dir: Path to the data directory.
        models_dir: Path to the models directory.
        base_version: Base version of the model.
        version: Version of the model.
        start_timestamp: Start timestamp.
        end_timestamp: End timestamp.
    """
    job_data_loader = DataLoader(data_dir=data_dir)
    job_model_manager = ModelManager(models_dir=models_dir)

    loaded_ok = job_model_manager.load(base_version)
    if not loaded_ok:
        return

    df = job_data_loader.load_window(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    if df.empty:
        return

    X, y = split_features_and_target(df)

    base_model = job_model_manager.model
    ml_task = job_model_manager.metadata["ml_task"]
    model_type = job_model_manager.metadata["model"]
    fit_kwargs = get_retraining_kwargs(model_type, base_model)
    model = clone(base_model)
    _ = train_model(model, model_type, X, y, **fit_kwargs)

    metadata = {
        "ml_task": ml_task,
        "version": version,
        "base_version": base_version,
        "model": model_type,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
    }

    job_model_manager.save(model, version, metadata)


class RetrainService:
    """Retraining service for predictor."""

    def __init__(self, data_dir: Path, model_manager: ModelManager):
        self._data_dir = data_dir
        self._model_manager = model_manager
        self._executor = ProcessPoolExecutor()
        self._jobs: dict[str, dict[str, RetrainStatus | str | Future]] = {}

    def start(self, start_timestamp: int, end_timestamp: int) -> str:
        """Start retraining for a given window.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            str: Job ID.
        """
        data_dir = self._data_dir
        models_dir = self._model_manager.models_dir
        base_version = self._model_manager.version
        version = self._model_manager.get_next_version()

        future: Future | None = self._executor.submit(
            _retrain_job,
            data_dir,
            models_dir,
            base_version,
            version,
            start_timestamp,
            end_timestamp,
        )

        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"status": RetrainStatus.RUNNING, "version": version, "future": future}

        def _on_done(future: Future) -> None:
            """Callback for when the future is done."""
            try:
                _ = future.result()
                loaded_ok = self._model_manager.load(version)
                self._jobs[job_id]["status"] = RetrainStatus.COMPLETED if loaded_ok else RetrainStatus.FAILED
            except Exception:
                self._jobs[job_id]["status"] = RetrainStatus.FAILED

        future.add_done_callback(_on_done)

        return job_id

    def get_status(self, job_id: str) -> RetrainStatus:
        """Get status of a retraining job.

        Args:
            job_id (str): Job ID.

        Returns:
            RetrainStatus: Status of the job.
        """
        status = self._jobs[job_id]["status"]
        return RetrainStatus(status)

    def clear(self) -> None:
        """Clear the retraining service."""
        for job in self._jobs.values():
            future: Future | None = job["future"]
            if future is not None and not future.done():
                future.cancel()

        self._jobs.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
