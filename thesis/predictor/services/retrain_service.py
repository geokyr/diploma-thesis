"""Retraining service for predictor."""

import logging
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from uuid import uuid4

from sklearn.base import clone

from thesis.common.config import (
    MAX_RETRIES_PREDICTOR,
    MAX_WORKERS_PREDICTOR,
    N_TRAINING_SAMPLES_ETA,
    SHRINK_FACTOR_ETA,
)
from thesis.common.enums import MLTask, ModelType, RetrainStatus
from thesis.common.schemas import RetrainResponse, RetrainStatusResponse
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_retraining_kwargs
from thesis.eta.pipeline import train_model
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.predictor import _TARGET_COLUMN_MAP

logger = logging.getLogger(__name__)


@dataclass
class RetrainJob:
    """
    Retraining job metadata.

    Attributes:
        status (RetrainStatus): Status of the job.
        version (str): Version of the model.
        base_version (str): Base version of the model.
        start_timestamp (int): Start timestamp of the job.
        end_timestamp (int): End timestamp of the job.
        retry_count (int): Number of retries of the job.
        future (Future | None): Future of the job.
    """

    status: RetrainStatus
    version: str
    base_version: str
    start_timestamp: int
    end_timestamp: int
    retry_count: int
    future: Future | None


# TODO: fuel and stops
N_TRAINING_SAMPLES_MAP: dict[MLTask, int] = {
    MLTask.ETA: N_TRAINING_SAMPLES_ETA,
    MLTask.FUEL: N_TRAINING_SAMPLES_ETA,
    MLTask.STOPS: N_TRAINING_SAMPLES_ETA,
}

# TODO: fuel and stops
SHRINK_FACTOR_MAP: dict[MLTask, float] = {
    MLTask.ETA: SHRINK_FACTOR_ETA,
    MLTask.FUEL: SHRINK_FACTOR_ETA,
    MLTask.STOPS: SHRINK_FACTOR_ETA,
}


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

    base_model = job_model_manager.model
    model_type = job_model_manager.metadata["model"]

    df = job_data_loader.load_window(start_timestamp=start_timestamp, end_timestamp=end_timestamp)
    if df.empty:
        return False

    X, y = split_features_and_target(df, target_columns=[_TARGET_COLUMN_MAP[ml_task]])
    X = X[base_model.feature_names_in_]

    n_training_samples = N_TRAINING_SAMPLES_MAP[ml_task]
    shrink_factor = SHRINK_FACTOR_MAP[ml_task]

    n_samples = len(df)
    data_percentage = n_samples / n_training_samples

    if model_type == ModelType.LIGHTGBM_REGRESSOR:
        original_n_estimators = base_model.booster_.num_trees()
    elif model_type == ModelType.XGBOOST_REGRESSOR:
        original_n_estimators = base_model.get_booster().num_boosted_rounds()

    new_n_estimators = int(original_n_estimators * data_percentage * shrink_factor)
    new_n_estimators = max(10, new_n_estimators)

    fit_kwargs = get_retraining_kwargs(model_type, base_model)
    model = clone(base_model).set_params(n_estimators=new_n_estimators)

    if model_type == ModelType.XGBOOST_REGRESSOR:
        model.set_params(early_stopping_rounds=None)

    _ = train_model(model, model_type, X, y, **fit_kwargs)

    metadata = {
        "ml_task": ml_task,
        "version": version,
        "base_version": base_version,
        "model": model_type,
        "start_timestamp": int(start_timestamp),
        "end_timestamp": int(end_timestamp),
        "n_estimators": new_n_estimators,
    }

    job_model_manager.save(model, version, metadata)

    return True


class RetrainService:
    """Retraining service for predictor."""

    def __init__(self, data_dir: Path, model_manager: ModelManager, ml_task: MLTask):
        self._data_dir = data_dir
        self._model_manager = model_manager
        self._ml_task = ml_task
        self._max_retries = MAX_RETRIES_PREDICTOR
        self._executor = ProcessPoolExecutor(max_workers=MAX_WORKERS_PREDICTOR)
        self._jobs: dict[str, RetrainJob] = {}

    def start(self, start_timestamp: int, end_timestamp: int) -> RetrainResponse:
        """
        Start retraining for a given window.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            RetrainResponse: Response containing job ID.
        """
        job_id = str(uuid4())
        base_version = self._model_manager.version
        version = self._model_manager.get_next_version()

        self._jobs[job_id] = RetrainJob(
            status=RetrainStatus.RUNNING,
            version=version,
            base_version=base_version,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            retry_count=0,
            future=None,
        )

        self._submit_job(job_id)

        return RetrainResponse(job_id=job_id)

    def _submit_job(self, job_id: str) -> None:
        """
        Submit or resubmit a retraining job.

        Args:
            job_id (str): Job ID.
        """
        job = self._jobs[job_id]

        future: Future = self._executor.submit(
            _retrain_job,
            self._data_dir,
            self._model_manager.models_dir,
            self._ml_task,
            job.base_version,
            job.version,
            job.start_timestamp,
            job.end_timestamp,
        )

        job.future = future
        future.add_done_callback(partial(self._on_retrain_done, job_id))

    def _on_retrain_done(self, job_id: str, future: Future) -> None:
        """
        Callback for when a retraining job is done.

        Args:
            job_id (str): Job ID.
            future (Future): Completed future.
        """
        job = self._jobs[job_id]

        try:
            success = future.result()
            loaded_ok = self._model_manager.load(job.version)

            if loaded_ok and success:
                job.status = RetrainStatus.COMPLETED
            else:
                self._handle_retrain_failure(job_id, error_msg="Job failed")
        except Exception as e:
            self._handle_retrain_failure(job_id, error_msg=f"Exception {e}")

    def _handle_retrain_failure(self, job_id: str, error_msg: str) -> None:
        """
        Handle retraining failure and retry if attempts remain.

        Args:
            job_id (str): Job ID.
            error_msg (str): Error message describing the failure.
        """
        job = self._jobs[job_id]

        if job.retry_count < self._max_retries:
            job.retry_count += 1
            self._submit_job(job_id)
        else:
            logger.error(f"Retraining job {job_id} failed after {self._max_retries} retries: {error_msg}")
            job.status = RetrainStatus.FAILED

    def get_status(self, job_id: str) -> RetrainStatusResponse:
        """
        Get status of a retraining job.

        Args:
            job_id (str): Job ID.

        Returns:
            RetrainStatusResponse: Status of the retraining job.
        """
        if job_id not in self._jobs:
            return RetrainStatusResponse(status=RetrainStatus.FAILED)

        job = self._jobs[job_id]
        return RetrainStatusResponse(status=job.status)

    def clear(self) -> None:
        """Clear the retraining service."""
        futures_to_cancel = []
        for job in self._jobs.values():
            if job.future is not None and not job.future.done():
                futures_to_cancel.append(job.future)

        for future in futures_to_cancel:
            future.cancel()

        self._jobs.clear()

        self._executor.shutdown(wait=False, cancel_futures=True)
