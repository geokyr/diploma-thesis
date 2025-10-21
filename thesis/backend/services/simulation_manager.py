"""Simulation manager for the simulation."""

import asyncio
import contextlib

import httpx

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.notification_store import NotificationStore
from thesis.backend.services.report_store import ReportStore
from thesis.backend.services.timelapse_driver import TimelapseDriver
from thesis.common.config import (
    ASYNC_CLIENT_TIMEOUT_SECONDS,
    MAX_RETRIES,
    PAUSE_POLL_SECONDS,
    RETRY_DELAY_SECONDS,
)
from thesis.common.enums import ReportStatus, SimulationState
from thesis.common.schemas import ReportGenerationRequest, ReportGenerationResponse, SimulationSnapshot


class SimulationManager:
    """Simulation manager for the simulation."""

    def __init__(
        self,
        timelapse_driver: TimelapseDriver,
        metrics_store: MetricsStore,
        notification_store: NotificationStore,
        report_store: ReportStore,
        summarizer_url: str,
    ) -> None:
        self._timelapse_driver: TimelapseDriver = timelapse_driver
        self._metrics_store: MetricsStore = metrics_store
        self._notification_store: NotificationStore = notification_store
        self._report_store: ReportStore = report_store
        self._summarizer_url: str = summarizer_url
        self._state: SimulationState = SimulationState.READY
        self._pause_poll_seconds: float = PAUSE_POLL_SECONDS
        self._tick_task: asyncio.Task | None = None

    async def _trigger_report_generation(self) -> None:
        """Trigger AI report generation with automatic retry on failure."""
        try:
            current_report_response = await self._report_store.get_report()
            if current_report_response.status in (ReportStatus.GENERATING, ReportStatus.READY):
                return

            await self._report_store.set_generating()

            notification_feed = await self._notification_store.get_all()

            ml_tasks = self._timelapse_driver.ml_tasks
            metrics = {}
            for ml_task in ml_tasks:
                metrics[ml_task] = await self._metrics_store.get_metrics(ml_task)

            report_request = ReportGenerationRequest(notifications=notification_feed.notifications, metrics=metrics)

            for attempt in range(MAX_RETRIES):
                try:
                    async with httpx.AsyncClient(timeout=ASYNC_CLIENT_TIMEOUT_SECONDS) as client:
                        url = f"{self._summarizer_url}/report/generate"
                        payload = report_request.model_dump(mode="json")
                        response = await client.post(url, json=payload)
                        response.raise_for_status()

                        report_response = ReportGenerationResponse.model_validate(response.json())

                    await self._report_store.set_ready(report_response.content)
                    return
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                    else:
                        raise

        except Exception:
            await self._report_store.set_failed()

    async def _tick_loop(self) -> None:
        """Tick loop for the simulation."""
        try:
            while True:
                state = self._state

                if state == SimulationState.RUNNING:
                    try:
                        should_continue = await self._timelapse_driver.run_tick()
                        if not should_continue:
                            self._state = SimulationState.COMPLETED
                            try:
                                await self._trigger_report_generation()
                            except Exception:
                                pass
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
                    await asyncio.sleep(self._timelapse_driver.interval_seconds)
                else:
                    await asyncio.sleep(self._pause_poll_seconds)
        except asyncio.CancelledError:
            return

    async def get_snapshot(self) -> SimulationSnapshot:
        """
        Get the snapshot of the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        state = self._state
        clock, drift_info = await self._timelapse_driver.get_snapshot_data()

        return SimulationSnapshot(state=state, clock=clock, drift_info=drift_info)

    async def start(self) -> SimulationSnapshot:
        """
        Start the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        if self._tick_task is None or self._tick_task.done():
            self._tick_task = asyncio.create_task(self._tick_loop())

        self._state = SimulationState.RUNNING

        return await self.get_snapshot()

    async def pause(self) -> SimulationSnapshot:
        """
        Pause the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        if self._state == SimulationState.RUNNING:
            self._state = SimulationState.PAUSED

        return await self.get_snapshot()

    async def resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        if self._state == SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

        return await self.get_snapshot()

    async def reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        self._state = SimulationState.READY
        task = self._tick_task
        self._tick_task = None

        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        try:
            await self._timelapse_driver.reset()
        except Exception:
            pass

        try:
            await self._report_store.reset()
        except Exception:
            pass

        return await self.get_snapshot()

    async def clear(self) -> None:
        """Clear the simulation."""
        self._state = SimulationState.READY
        task = self._tick_task
        self._tick_task = None

        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        try:
            await self._timelapse_driver.clear()
        except Exception:
            pass

        try:
            await self._report_store.reset()
        except Exception:
            pass
