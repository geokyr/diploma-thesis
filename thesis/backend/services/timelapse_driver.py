from dataclasses import dataclass

import httpx

from thesis.backend.services.metrics_store import MetricsStore
from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS, INTERVAL_MS, SPEED_MULTIPLIER
from thesis.common.schemas import PredictionBatchResponse
from thesis.common.service import PlatformServiceConfig


@dataclass(slots=True)
class SimulationClock:
    speed_multiplier: float = SPEED_MULTIPLIER
    time: int = 0

    def tick(self, real_seconds: float = 1.0) -> tuple[float, float]:
        start_timestamp = self.time
        self.time += real_seconds * self.speed_multiplier
        end_timestamp = self.time

        return start_timestamp, end_timestamp


class TimelapseDriver:
    def __init__(
        self, metrics_store: MetricsStore | None = None, service_config: PlatformServiceConfig | None = None
    ) -> None:
        self._metrics_store = metrics_store or MetricsStore()
        self._cfg = service_config or PlatformServiceConfig()
        self._clock = SimulationClock()

    async def _predict_eta(self, start_ts: float, end_ts: float) -> PredictionBatchResponse | None:
        url = f"{self._cfg.predictor_eta_url}/predict/batch"
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as client:
            try:
                resp = await client.post(url, json={"start_timestamp": start_ts, "end_timestamp": end_ts})
                resp.raise_for_status()
                data = resp.json()
                return PredictionBatchResponse(**data)
            except Exception:
                return None

    async def _forward_to_drift(self, start_ts: float, end_ts: float, batch: PredictionBatchResponse) -> None:
        url = f"{self._cfg.drift_url}/drift/errors"
        payload = {
            "task": "eta",
            "points": [{"timestamp": ep.timestamp, "error": ep.error} for ep in (batch.error_points or [])],
        }
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as client:
            try:
                await client.post(url, json=payload)
            except Exception:
                pass

    async def run_tick(self) -> None:
        start_timestamp, end_timestamp = self._clock.tick(real_seconds=INTERVAL_MS / 1000.0)
        batch = await self._predict_eta(start_timestamp, end_timestamp)
        if batch is None:
            return
        await self._forward_to_drift(start_timestamp, end_timestamp, batch)
        self._metrics_store.push(end_timestamp, batch.mae)

    @property
    def time(self) -> float:
        return self._clock.time

    def restart(self) -> None:
        self._clock.time = 0.0

    # TODO: refactor
    async def clear(self) -> None: ...
