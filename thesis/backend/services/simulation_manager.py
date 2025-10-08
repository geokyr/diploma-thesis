"""Simulation manager for the simulation."""

import asyncio
import contextlib

from thesis.backend.services.timelapse_driver import TimelapseDriver
from thesis.common.config import PAUSE_POLL_SECONDS
from thesis.common.enums import SimulationState
from thesis.common.schemas import SimulationSnapshot


class SimulationManager:
    """Simulation manager for the simulation."""

    def __init__(self, timelapse_driver: TimelapseDriver) -> None:
        self._timelapse_driver: TimelapseDriver = timelapse_driver
        self._state: SimulationState = SimulationState.IDLE
        self._pause_poll_seconds: float = PAUSE_POLL_SECONDS
        self._tick_task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def _tick_loop(self) -> None:
        """Tick loop for the simulation."""
        try:
            while True:
                state = self._state
                if state == SimulationState.RUNNING:
                    try:
                        should_continue = await self._timelapse_driver.run_tick()
                        if not should_continue:
                            async with self._lock:
                                self._state = SimulationState.PAUSED
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
        async with self._lock:
            state = self._state
            clock = self._timelapse_driver.clock
            drift_info = self._timelapse_driver.drift_info

        return SimulationSnapshot(state=state, clock=clock, drift_info=drift_info)

    async def start(self) -> SimulationSnapshot:
        """
        Start the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        async with self._lock:
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
        async with self._lock:
            if self._state == SimulationState.RUNNING:
                self._state = SimulationState.PAUSED

        return await self.get_snapshot()

    async def resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        async with self._lock:
            if self._state == SimulationState.PAUSED:
                self._state = SimulationState.RUNNING

        return await self.get_snapshot()

    async def reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        async with self._lock:
            self._state = SimulationState.IDLE
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

        return await self.get_snapshot()

    async def clear(self) -> None:
        """Clear the simulation."""
        async with self._lock:
            self._state = SimulationState.IDLE
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
