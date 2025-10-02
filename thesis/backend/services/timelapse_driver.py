"""Timelapse driver for the simulation."""

import asyncio
import contextlib

import httpx

from thesis.backend.services.metrics_store import MetricsStore
from thesis.common.config import COLLECT_SECONDS, HTTP_CLIENT_TIMEOUT_SECONDS, INTERVAL_SECONDS, SPEED_MULTIPLIER
from thesis.common.enums import DriftState, MLTask, RetrainStatus
from thesis.common.schemas import (
    DriftErrorsRequest,
    DriftErrorsResponse,
    DriftInfo,
    ErrorPoint,
    PredictionBatchRequest,
    PredictionBatchResponse,
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

    def __init__(self, config: PlatformServiceConfig, metrics_store: MetricsStore) -> None:
        self._config: PlatformServiceConfig = config
        self._metrics_store: MetricsStore = metrics_store
        self._speed_multiplier: int = SPEED_MULTIPLIER
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS)
        self._tick_lock: asyncio.Lock = asyncio.Lock()
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
            ml_task: DriftInfo(state=DriftState.STABLE, start_timestamp=None, collecting=False, job_id=None)
            for ml_task in self.ml_tasks
        }

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

    async def _start_retrain(self, task: MLTask, start_timestamp: int, end_timestamp: int) -> RetrainResponse | None:
        """
        Start retraining for a given ML task.

        Args:
            task (MLTask): ML task.
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            RetrainResult | None: Retrain result or None if failed to start retraining.
        """
        url = f"{self.predictor_urls[task]}/retrain/start"
        payload = RetrainRequest(start_timestamp=start_timestamp, end_timestamp=end_timestamp).model_dump()
        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return RetrainResponse.model_validate(response.json())
        except Exception:
            return None

    async def _poll_retrain(self, task: MLTask, job_id: str) -> RetrainStatus:
        """
        Poll the retrain status for a given ML task and job ID.

        Args:
            task (MLTask): ML task.
            job_id (str): Job ID.

        Returns:
            RetrainStatus: Status of the retrain.
        """
        predictor_url = self.predictor_urls[task]
        url = f"{predictor_url}/retrain/status/{job_id}"
        try:
            while True:
                response = await self._client.get(url)
                response.raise_for_status()
                status = RetrainStatusResponse.model_validate(response.json()).status

                if status in (RetrainStatus.COMPLETED, RetrainStatus.FAILED):
                    return status

                await asyncio.sleep(self.interval_seconds)
        except Exception:
            return RetrainStatus.FAILED

    async def _handle_retrain(self, ml_task: MLTask, job_id: str) -> None:
        """
        Handle the retrain for a given ML task.

        Args:
            ml_task (MLTask): ML task.
            job_id (str): Job ID.
        """
        info = self.drift_info[ml_task]
        info.job_id = job_id

        status = await self._poll_retrain(ml_task, job_id)

        if status == RetrainStatus.COMPLETED:
            info.state = DriftState.STABLE
        elif status == RetrainStatus.FAILED:
            info.state = DriftState.DRIFTED

        info.start_timestamp = None
        info.collecting = False
        info.job_id = None

    async def run_tick(self) -> None:
        """Run a tick of the simulation."""
        async with self._tick_lock:
            start_timestamp, end_timestamp = self._advance_clock()

            for ml_task in self.ml_tasks:
                try:
                    batch = await self._predict_window(ml_task, start_timestamp, end_timestamp)
                except Exception:
                    continue

                if batch.mae is not None:
                    await self._metrics_store.push(ml_task, end_timestamp, batch.mae)

                try:
                    drift_response = await self._check_for_drift(ml_task, batch.error_points)
                except Exception:
                    continue

                if drift_response is None:
                    continue

                info = self.drift_info[ml_task]

                if drift_response.state == DriftState.DRIFTED and not info.collecting:
                    info.state = DriftState.DRIFTED
                    info.start_timestamp = end_timestamp
                    info.collecting = True

                if info.collecting and info.start_timestamp is not None:
                    data_collected = (end_timestamp - info.start_timestamp) >= self._collect_seconds
                    if data_collected and info.job_id is None:
                        info.state = DriftState.RETRAINING
                        start_timestamp = info.start_timestamp - self._collect_seconds
                        if start_timestamp < 0:
                            start_timestamp = 0
                        retrain_result = await self._start_retrain(ml_task, start_timestamp, end_timestamp)
                        job_id = retrain_result.job_id if retrain_result else None
                        if job_id:
                            task = asyncio.create_task(self._handle_retrain(ml_task, job_id))
                            self._retrain_tasks.append(task)
                        else:
                            info.state = DriftState.DRIFTED
                            info.start_timestamp = None
                            info.collecting = False
                            info.job_id = None

    async def reset(self) -> None:
        """Reset the timelapse driver."""
        for task in self._retrain_tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._retrain_tasks.clear()

        await self._metrics_store.reset()
        self.clock = 0
        self._reset_drift_info()

    async def clear(self) -> None:
        """Clear the timelapse driver."""
        for task in self._retrain_tasks:
            if task and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._retrain_tasks.clear()

        await self._metrics_store.clear()
        self.clock = 0
        self._reset_drift_info()
        await self._client.aclose()
