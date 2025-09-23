"""Timelapse driver for the simulation."""

import asyncio

import httpx

from thesis.backend.services.metrics_store import MetricsStore
from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS, INTERVAL_SECONDS, SPEED_MULTIPLIER
from thesis.common.enums import DriftState
from thesis.common.schemas import (
    DriftErrorsRequest,
    DriftErrorsResponse,
    ErrorPoint,
    PredictionBatchRequest,
    PredictionBatchResponse,
)
from thesis.common.service import PlatformServiceConfig


class TimelapseDriver:
    """
    Timelapse driver for the simulation.

    Attributes:
        clock (int): The current timelapse clock time.
    """

    def __init__(self, config: PlatformServiceConfig, metrics_store: MetricsStore) -> None:
        self._config: PlatformServiceConfig = config
        self._metrics_store: MetricsStore = metrics_store
        self._speed_multiplier: int = SPEED_MULTIPLIER
        self._interval_seconds: int = INTERVAL_SECONDS
        self._client: httpx.AsyncClient = httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS)
        self._tick_lock: asyncio.Lock = asyncio.Lock()
        self.clock: int = 0

    def _advance_clock(self) -> tuple[int, int]:
        """
        Advance the clock by the interval seconds multiplied by the speed multiplier.

        Returns:
            tuple[int, int]: The start and end timestamps.
        """
        start_timestamp = self.clock
        self.clock += self._interval_seconds * self._speed_multiplier
        end_timestamp = self.clock

        return start_timestamp, end_timestamp

    async def _predict_eta(self, start_timestamp: int, end_timestamp: int) -> PredictionBatchResponse:
        """
        Predict the ETA for a given time window.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            PredictionBatchResponse: Prediction batch response.
        """
        url = f"{self._config.predictor_eta_url}/predict/batch"
        payload = PredictionBatchRequest(start_timestamp=start_timestamp, end_timestamp=end_timestamp)

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return PredictionBatchResponse(**response.json())
        except Exception:
            return PredictionBatchResponse(error_points=[], mae=None)

    async def _check_for_drift(self, error_points: list[ErrorPoint]) -> DriftErrorsResponse | None:
        """
        Check for drift in the error points.

        Args:
            error_points (list[ErrorPoint]): List of error points.

        Returns:
            DriftErrorsResponse | None: Drift errors response or None if failed to check for drift.
        """
        url = f"{self._config.drift_url}/drift/errors"
        payload = DriftErrorsRequest(ml_task=self._config.ml_task, error_points=error_points)

        try:
            response = await self._client.post(url, json=payload)
            response.raise_for_status()
            return DriftErrorsResponse(**response.json())
        except Exception:
            return None

    async def run_tick(self) -> None:
        """Run a tick of the simulation."""
        async with self._tick_lock:
            start_timestamp, end_timestamp = self._advance_clock()

            try:
                batch = await self._predict_eta(start_timestamp, end_timestamp)
            except Exception:
                return

            if batch.mae is not None:
                await self._metrics_store.push(end_timestamp, batch.mae)

            try:
                drift_response = await self._check_for_drift(batch.error_points)
            except Exception:
                return

            if drift_response is None:
                return

            if drift_response.state == DriftState.DRIFTED:
                # TODO: handle drift
                pass

    async def reset(self) -> None:
        """Reset the timelapse driver."""
        await self._metrics_store.reset()
        self.clock = 0

    async def clear(self) -> None:
        """Clear the timelapse driver."""
        await self._metrics_store.clear()
        self.clock = 0
        await self._client.aclose()
