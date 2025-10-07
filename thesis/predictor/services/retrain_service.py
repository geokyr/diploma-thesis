"""Retraining service for predictor."""

from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

from sklearn.base import clone

from thesis.common.enums import RetrainStatus
from thesis.common.schemas import RetrainResponse, RetrainStatusResponse
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_retraining_kwargs
from thesis.eta.pipeline import compute_absolute_errors, train_model
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager


def _retrain_job(
    data_dir: Path, models_dir: Path, base_version: str, version: str, start_timestamp: int, end_timestamp: int
) -> list[float] | None:
    """
    Retrain job.

    Args:
        data_dir: Path to the data directory.
        models_dir: Path to the models directory.
        base_version: Base version of the model.
        version: Version of the model.
        start_timestamp: Start timestamp.
        end_timestamp: End timestamp.

    Returns:
        list[float] | None: Post-adaptation absolute errors on training data.
    """
    job_data_loader = DataLoader(data_dir=data_dir)
    job_model_manager = ModelManager(models_dir=models_dir)

    loaded_ok = job_model_manager.load(base_version)
    if not loaded_ok:
        return None

    df = job_data_loader.load_window(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    if df.empty:
        return None

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

    y_pred = model.predict(X)
    absolute_errors = compute_absolute_errors(y.values, y_pred).tolist()

    return absolute_errors


class RetrainService:
    """Retraining service for predictor."""

    def __init__(self, data_dir: Path, model_manager: ModelManager):
        self._data_dir = data_dir
        self._model_manager = model_manager
        self._executor = ProcessPoolExecutor()
        self._jobs: dict[str, dict[str, RetrainStatus | str | Future | list[float] | None]] = {}

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

        job_id = str(uuid4())
        self._jobs[job_id] = {
            "status": RetrainStatus.RUNNING,
            "version": version,
            "future": future,
            "post_adaptation_errors": None,
        }

        def _on_done(future: Future) -> None:
            """
            Callback for when the future is done.

            Args:
                future (Future): Future.
            """
            try:
                post_adaptation_errors = future.result()
                loaded_ok = self._model_manager.load(version)
                if loaded_ok and post_adaptation_errors is not None:
                    self._jobs[job_id]["status"] = RetrainStatus.COMPLETED
                    self._jobs[job_id]["post_adaptation_errors"] = post_adaptation_errors
                else:
                    self._jobs[job_id]["status"] = RetrainStatus.FAILED
            except Exception:
                self._jobs[job_id]["status"] = RetrainStatus.FAILED

        future.add_done_callback(_on_done)

        return RetrainResponse(job_id=job_id)

    def get_status(self, job_id: str) -> RetrainStatusResponse:
        """
        Get status and post-adaptation errors of a retraining job.

        Args:
            job_id (str): Job ID.

        Returns:
            RetrainStatusResponse: Status and post-adaptation errors.
        """
        if job_id not in self._jobs:
            return RetrainStatusResponse(status=RetrainStatus.FAILED, post_adaptation_errors=None)

        job = self._jobs[job_id]
        status = RetrainStatus(job["status"])
        post_adaptation_errors = job.get("post_adaptation_errors") if status == RetrainStatus.COMPLETED else None
        return RetrainStatusResponse(status=status, post_adaptation_errors=post_adaptation_errors)

    def clear(self) -> None:
        """Clear the retraining service."""
        for job in self._jobs.values():
            future: Future | None = job["future"]
            if future is not None and not future.done():
                future.cancel()

        self._jobs.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
