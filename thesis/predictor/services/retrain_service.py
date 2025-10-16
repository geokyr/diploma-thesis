"""Retraining service for predictor."""

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from threading import Lock
from uuid import uuid4

from sklearn.base import clone

from thesis.common.enums import MLTask, RetrainStatus
from thesis.common.schemas import RetrainResponse, RetrainStatusResponse
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_retraining_kwargs
from thesis.eta.pipeline import train_model
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.predictor import _TARGET_COLUMN_MAP


def _retrain_job(
    data_dir: Path,
    models_dir: Path,
    ml_task: MLTask,
    base_version: str,
    version: str,
    start_timestamp: int,
    end_timestamp: int,
) -> bool:
    """
    Retrain job.

    Args:
        data_dir (Path): Path to the data directory.
        models_dir (Path): Path to the models directory.
        ml_task (MLTask): ML task type.
        base_version (str): Base version of the model.
        version (str): Version of the model.
        start_timestamp (int): Start timestamp.
        end_timestamp (int): End timestamp.

    Returns:
        bool: True if retraining succeeded, False otherwise.
    """
    job_model_manager = ModelManager(models_dir=models_dir)
    job_data_loader = DataLoader(data_dir=data_dir, ml_task=ml_task)

    loaded_ok = job_model_manager.load(base_version)
    if not loaded_ok:
        return False

    df = job_data_loader.load_window(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    if df.empty:
        return False

    X, y = split_features_and_target(df, target_columns=[_TARGET_COLUMN_MAP[ml_task]])

    base_model = job_model_manager.model
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

    return True


class RetrainService:
    """Retraining service for predictor."""

    def __init__(self, data_dir: Path, model_manager: ModelManager, ml_task: MLTask):
        self._data_dir = data_dir
        self._model_manager = model_manager
        self._ml_task = ml_task
        self._executor = ProcessPoolExecutor()
        self._jobs: dict[str, dict[str, RetrainStatus | str | Future]] = {}
        self._jobs_lock = Lock()

    def start(self, start_timestamp: int, end_timestamp: int) -> RetrainResponse:
        """
        Start retraining for a given window.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            RetrainResponse: Response containing job ID.
        """
        data_dir = self._data_dir
        models_dir = self._model_manager.models_dir
        ml_task = self._ml_task
        base_version = self._model_manager.version
        version = self._model_manager.get_next_version()

        future: Future | None = self._executor.submit(
            _retrain_job, data_dir, models_dir, ml_task, base_version, version, start_timestamp, end_timestamp
        )

        job_id = str(uuid4())

        with self._jobs_lock:
            self._jobs[job_id] = {
                "status": RetrainStatus.RUNNING,
                "version": version,
                "future": future,
            }

        def _on_done(future: Future) -> None:
            """
            Callback for when the future is done.

            Args:
                future (Future): Future.
            """
            try:
                success = future.result()
                loaded_ok = self._model_manager.load(version)

                with self._jobs_lock:
                    if loaded_ok and success:
                        self._jobs[job_id]["status"] = RetrainStatus.COMPLETED
                    else:
                        self._jobs[job_id]["status"] = RetrainStatus.FAILED
            except Exception:
                with self._jobs_lock:
                    self._jobs[job_id]["status"] = RetrainStatus.FAILED

        future.add_done_callback(_on_done)

        return RetrainResponse(job_id=job_id)

    def get_status(self, job_id: str) -> RetrainStatusResponse:
        """
        Get status of a retraining job.

        Args:
            job_id (str): Job ID.

        Returns:
            RetrainStatusResponse: Status of the retraining job.
        """
        with self._jobs_lock:
            if job_id not in self._jobs:
                return RetrainStatusResponse(status=RetrainStatus.FAILED)

            job = self._jobs[job_id]
            status = RetrainStatus(job["status"])
            return RetrainStatusResponse(status=status)

    def clear(self) -> None:
        """Clear the retraining service."""
        with self._jobs_lock:
            futures_to_cancel = []
            for job in self._jobs.values():
                future: Future | None = job["future"]
                if future is not None and not future.done():
                    futures_to_cancel.append(future)

        for future in futures_to_cancel:
            future.cancel()

        with self._jobs_lock:
            self._jobs.clear()

        self._executor.shutdown(wait=False, cancel_futures=True)
