"""Timelapse driver for the simulation."""

import asyncio
import contextlib

import httpx

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.notification_store import NotificationStore
from thesis.common.config import COLLECT_SECONDS, HTTP_CLIENT_TIMEOUT_SECONDS, INTERVAL_SECONDS, SPEED_MULTIPLIER
from thesis.common.enums import DriftState, MLTask, NotificationLevel, RetrainStatus
from thesis.common.schemas import (
    DriftErrorsRequest,
    DriftErrorsResponse,
    DriftInfo,
    DriftResetRequest,
    DriftResetResponse,
    ErrorPoint,
    PredictionBatchRequest,
    PredictionBatchResponse,
    RecalibrateRequest,
    RecalibrateResponse,
    RetrainRequest,
    RetrainResponse,
    RetrainStatusResponse,
)
from thesis.common.service import PlatformServiceConfig


class TimelapseDriver:
    """
    Timelapse driver for the simulation.

    Attributes:
        clock (int): The current timelapse clock time.
        interval_seconds (int): The interval seconds.
        drift_info (dict[MLTask, DriftInfo]): The drift info per ML task.
        ml_tasks (list[MLTask]): The ML tasks.
        predictor_urls (dict[MLTask, str]): The predictor URLs.
    """

    def __init__(
        self, config: PlatformServiceConfig, metrics_store: MetricsStore, notification_store: NotificationStore
    ) -> None:
        self._config: PlatformServiceConfig = config
        self._metrics_store: MetricsStore = metrics_store
        self._notification_store: NotificationStore = notification_store
        self._speed_multiplier: int = SPEED_MULTIPLIER
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS)
        self._clock_lock: asyncio.Lock = asyncio.Lock()
        self._tasks_lock: asyncio.Lock = asyncio.Lock()
        self._drift_info_locks: dict[MLTask, asyncio.Lock] = {}
        self._collect_seconds: int = COLLECT_SECONDS
        self._retrain_tasks: list[asyncio.Task] = []
        self.clock: int = 0
        self.interval_seconds: int = INTERVAL_SECONDS
        self.drift_info: dict[MLTask, DriftInfo] = {}
        self.ml_tasks: list[MLTask] = []
        self.predictor_urls: dict[MLTask, str] = {
            MLTask.ETA: self._config.predictor_eta_url,
            MLTask.FUEL: self._config.predictor_fuel_url,
            MLTask.STOPS: self._config.predictor_stops_url,
        }

    def _reset_drift_info(self) -> None:
        """Reset the drift info."""
        self.drift_info = {
            ml_task: DriftInfo(state=DriftState.CALIBRATING, start_timestamp=None, collecting=False, job_id=None)
            for ml_task in self.ml_tasks
        }

    async def get_snapshot_data(self) -> tuple[int, dict[MLTask, DriftInfo]]:
        """
        Get snapshot of clock and drift info atomically.

        Returns:
            tuple[int, dict[MLTask, DriftInfo]]: Clock and drift info snapshot.
        """
        async with self._clock_lock:
            clock = self.clock
            drift_info_snapshot = dict(self.drift_info)

        return clock, drift_info_snapshot

    async def _check_ml_task_availability(self, ml_task: MLTask, url: str) -> tuple[MLTask, bool]:
        """
        Check if the ML task is available.

        Args:
            ml_task (MLTask): ML task.
            url (str): URL of the predictor.

        Returns:
            tuple[MLTask, bool]: ML task and availability.
        """
        try:
            response = await self._client.get(f"{url}/health")
            return ml_task, response.status_code == 200
        except Exception:
            return ml_task, False

    async def detect_available_ml_tasks(self) -> None:
        """Detect available ML tasks."""
        self.ml_tasks.clear()

        results = await asyncio.gather(
            *(self._check_ml_task_availability(ml_task, url) for ml_task, url in self.predictor_urls.items())
        )

        self.ml_tasks = [ml_task for ml_task, available in results if available]
        self._drift_info_locks = {ml_task: asyncio.Lock() for ml_task in self.ml_tasks}

        self._reset_drift_info()

    def _advance_clock(self) -> tuple[int, int]:
        """
        Advance the clock by the interval seconds multiplied by the speed multiplier.

        Returns:
            tuple[int, int]: The start and end timestamps.
        """
        start_timestamp = self.clock
        self.clock += self.interval_seconds * self._speed_multiplier
        end_timestamp = self.clock

        return start_timestamp, end_timestamp

    async def _predict_window(
        self, ml_task: MLTask, start_timestamp: int, end_timestamp: int
    ) -> PredictionBatchResponse:
        """
        Predict the given time window for a given ML task.

        Args:
            ml_task (MLTask): ML task.
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            PredictionBatchResponse: Prediction batch response.
        """
        url = f"{self.predictor_urls[ml_task]}/predict/batch"
        payload = PredictionBatchRequest(start_timestamp=start_timestamp, end_timestamp=end_timestamp).model_dump()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return PredictionBatchResponse.model_validate(response.json())
        except Exception:
            return PredictionBatchResponse(error_points=[], mae=None)

    async def _check_for_drift(self, ml_task: MLTask, error_points: list[ErrorPoint]) -> DriftErrorsResponse | None:
        """
        Check for drift in the error points.

        Args:
            ml_task (MLTask): ML task.
            error_points (list[ErrorPoint]): List of error points.

        Returns:
            DriftErrorsResponse | None: Drift errors response or None if failed to check for drift.
        """
        url = f"{self._config.drift_url}/drift/errors"
        payload = DriftErrorsRequest(ml_task=ml_task, error_points=error_points).model_dump()

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return DriftErrorsResponse.model_validate(response.json())
        except Exception:
            return None

    async def _start_retrain(self, task: MLTask, start_timestamp: int, end_timestamp: int) -> str | None:
        """
        Start retraining for a given ML task.

        Args:
            task (MLTask): ML task.
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            str | None: Job ID or None if failed to start retraining.
        """
        url = f"{self.predictor_urls[task]}/retrain/start"
        payload = RetrainRequest(start_timestamp=start_timestamp, end_timestamp=end_timestamp).model_dump()
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            retrain_response = RetrainResponse.model_validate(response.json())
            return retrain_response.job_id
        except Exception:
            return None

    async def _poll_retrain(self, task: MLTask, job_id: str) -> RetrainStatus:
        """
        Poll the retrain status for a given ML task and job ID.

        Args:
            task (MLTask): ML task.
            job_id (str): Job ID.

        Returns:
            RetrainStatus: Status of the retrain job.
        """
        url = f"{self.predictor_urls[task]}/retrain/status/{job_id}"
        try:
            while True:
                response = await self._client.get(url)
                response.raise_for_status()
                status_response = RetrainStatusResponse.model_validate(response.json())

                if status_response.status in (RetrainStatus.COMPLETED, RetrainStatus.FAILED):
                    return status_response.status

                await asyncio.sleep(self.interval_seconds)
        except Exception:
            return RetrainStatus.FAILED

    async def _recalibrate_drift_detectors(self, ml_task: MLTask) -> bool:
        """
        Recalibrate drift detectors after model adaptation.

        Args:
            ml_task (MLTask): ML task.

        Returns:
            bool: True if successful, False otherwise.
        """
        url = f"{self._config.drift_url}/drift/recalibrate"
        payload = RecalibrateRequest(ml_task=ml_task).model_dump()
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            recalibrate_response = RecalibrateResponse.model_validate(response.json())
            return recalibrate_response.success
        except Exception:
            return False

    async def _handle_retrain(self, ml_task: MLTask, job_id: str) -> None:
        """
        Handle the complete retrain pipeline for a given ML task.

        Args:
            ml_task (MLTask): ML task.
            job_id (str): Job ID.
        """
        lock = self._drift_info_locks[ml_task]

        async with lock:
            self.drift_info[ml_task].job_id = job_id

        status = await self._poll_retrain(ml_task, job_id)

        if status == RetrainStatus.COMPLETED:
            success = await self._recalibrate_drift_detectors(ml_task)

            async with lock:
                info = self.drift_info[ml_task]
                info.state = DriftState.CALIBRATING if success else DriftState.DRIFTED
                info.start_timestamp = None
                info.collecting = False
                info.job_id = None

            if success:
                await self._notification_store.push(
                    self.clock, "Model updated, collecting calibration data", NotificationLevel.SUCCESS, ml_task
                )
            else:
                await self._notification_store.push(
                    self.clock, "Recalibration failed", NotificationLevel.DANGER, ml_task
                )
        else:
            async with lock:
                info = self.drift_info[ml_task]
                info.state = DriftState.DRIFTED
                info.start_timestamp = None
                info.collecting = False
                info.job_id = None

            await self._notification_store.push(self.clock, "Retraining failed", NotificationLevel.DANGER, ml_task)

    async def run_tick(self) -> bool:
        """
        Run a tick of the simulation.

        Returns:
            bool: True if simulation should continue, False if data has ended.
        """
        current_clock = self.clock

        if current_clock == 0:
            await self._notification_store.push(
                current_clock, "First day started with normal conditions", NotificationLevel.SUCCESS
            )
        elif current_clock == 36000:
            await self._notification_store.push(
                current_clock, "Second day started with rain conditions", NotificationLevel.SUCCESS
            )
        elif current_clock == 72000:
            await self._notification_store.push(current_clock, "Data has been exhausted", NotificationLevel.SUCCESS)
            return False

        async with self._clock_lock:
            start_timestamp, end_timestamp = self._advance_clock()

        for ml_task in self.ml_tasks:
            lock = self._drift_info_locks[ml_task]

            try:
                batch = await self._predict_window(ml_task, start_timestamp, end_timestamp)
            except Exception:
                continue

            if batch.mae is not None and batch.error_points:
                n_samples = len(batch.error_points)
                await self._metrics_store.push(ml_task, end_timestamp, batch.mae, n_samples)

            try:
                drift_response = await self._check_for_drift(ml_task, batch.error_points)
            except Exception:
                continue

            if drift_response is None:
                continue

            notifications_to_push = []
            should_retrain = False
            retrain_start_timestamp = 0

            async with lock:
                info = self.drift_info[ml_task]

                if drift_response.state == DriftState.STABLE and info.state == DriftState.CALIBRATING:
                    info.state = DriftState.STABLE
                    notifications_to_push.append(
                        (
                            end_timestamp,
                            "Calibration complete, monitoring active",
                            NotificationLevel.SUCCESS,
                            ml_task,
                        )
                    )

                if (
                    drift_response.state == DriftState.DRIFTED
                    and info.state == DriftState.STABLE
                    and not info.collecting
                ):
                    info.state = DriftState.DRIFTED
                    info.start_timestamp = end_timestamp
                    info.collecting = True
                    notifications_to_push.append((end_timestamp, "Drift detected", NotificationLevel.DANGER, ml_task))

                if info.collecting and info.start_timestamp is not None:
                    data_collected = (end_timestamp - info.start_timestamp) >= self._collect_seconds
                    if data_collected and info.job_id is None:
                        should_retrain = True
                        info.state = DriftState.RETRAINING
                        retrain_start_timestamp = info.start_timestamp - self._collect_seconds
                        retrain_start_timestamp = 0 if retrain_start_timestamp < 0 else retrain_start_timestamp

            for notification in notifications_to_push:
                await self._notification_store.push(*notification)

            if should_retrain:
                await self._notification_store.push(
                    end_timestamp, "Retraining model", NotificationLevel.WARNING, ml_task
                )
                job_id = await self._start_retrain(ml_task, retrain_start_timestamp, end_timestamp)

                async with lock:
                    info = self.drift_info[ml_task]
                    if job_id:
                        task = asyncio.create_task(self._handle_retrain(ml_task, job_id))
                        async with self._tasks_lock:
                            self._retrain_tasks.append(task)
                    else:
                        info.state = DriftState.DRIFTED
                        info.start_timestamp = None
                        info.collecting = False
                        info.job_id = None

                if not job_id:
                    await self._notification_store.push(
                        end_timestamp, "Retraining failed", NotificationLevel.DANGER, ml_task
                    )

        return True

    async def _reset_drift_service(self) -> bool:
        """
        Reset the drift service for available ML tasks.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            url = f"{self._config.drift_url}/drift/reset"
            payload = DriftResetRequest(ml_tasks=self.ml_tasks)

            response = await self._client.post(url, json=payload.model_dump())
            response.raise_for_status()
            reset_response = DriftResetResponse.model_validate(response.json())

            return reset_response.success

        except Exception:
            return False

    async def reset(self) -> None:
        """Reset the timelapse driver."""
        async with self._tasks_lock:
            tasks_to_cancel = list(self._retrain_tasks)
            self._retrain_tasks.clear()

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self._metrics_store.reset()
        await self._notification_store.reset()

        async with self._clock_lock:
            self.clock = 0

        self._reset_drift_info()
        await self._reset_drift_service()

    async def clear(self) -> None:
        """Clear the timelapse driver."""
        async with self._tasks_lock:
            tasks_to_cancel = list(self._retrain_tasks)
            self._retrain_tasks.clear()

        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        await self._metrics_store.clear()
        await self._notification_store.clear()

        async with self._clock_lock:
            self.clock = 0

        self._reset_drift_info()

        await self._client.aclose()
