"""Drift detection service for ML tasks."""

from thesis.common.enums import MLTask
from thesis.common.schemas import DriftErrorsResponse, DriftResetResponse, ErrorPoint, RecalibrateResponse
from thesis.drift.utils.drift_worker import DriftWorker


class DriftService:
    """Drift detection service with workers for each ML task."""

    def __init__(self) -> None:
        self._workers: dict[MLTask, DriftWorker] = {}

    def _get_or_create_worker(self, ml_task: MLTask) -> DriftWorker:
        """
        Get existing worker or create and start a new one.

        Args:
            ml_task (MLTask): ML task to get worker for.

        Returns:
            DriftWorker: Worker for the task.
        """
        if ml_task not in self._workers:
            worker = DriftWorker(ml_task)
            worker.start()
            self._workers[ml_task] = worker
        return self._workers[ml_task]

    async def process_errors(self, ml_task: MLTask, error_points: list[ErrorPoint]) -> DriftErrorsResponse:
        """
        Process errors by routing to appropriate worker.

        Args:
            ml_task (MLTask): ML task to process errors for.
            error_points (list[ErrorPoint]): List of error points with timestamps.

        Returns:
            DriftErrorsResponse: Current drift state after processing errors.
        """
        worker = self._get_or_create_worker(ml_task)
        return await worker.submit(error_points)

    async def reset_tasks(self, ml_tasks: list[MLTask]) -> DriftResetResponse:
        """
        Reset drift detection for a list of ML tasks.

        Args:
            ml_tasks (list[MLTask]): List of ML tasks to reset.

        Returns:
            DriftResetResponse: Response containing success status.
        """
        success = True
        for ml_task in ml_tasks:
            if ml_task in self._workers:
                worker = self._workers[ml_task]
                task_success = await worker.reset()
                success = success and task_success

        return DriftResetResponse(success=success)

    async def recalibrate_task(self, ml_task: MLTask) -> RecalibrateResponse:
        """
        Recalibrate drift detectors after model adaptation.

        Args:
            ml_task (MLTask): ML task to recalibrate.

        Returns:
            RecalibrateResponse: Recalibration success status.
        """
        worker = self._get_or_create_worker(ml_task)
        success = await worker.reset()

        return RecalibrateResponse(success=success)

    async def clear(self) -> None:
        """Clear the drift service."""
        if not self._workers:
            return

        for worker in self._workers.values():
            worker.clear()
