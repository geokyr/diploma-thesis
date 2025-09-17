import asyncio
import contextlib
from dataclasses import dataclass
from typing import Optional

from thesis.backend.services.timelapse import TimelapseDriver
from thesis.common.enums import SimulationState
from thesis.common.schemas import SimulationStatus


@dataclass(slots=True)
class _SimState:
    state: SimulationState = SimulationState.IDLE
    current_sim_time: float = 0.0


class SimulationManager:
    def __init__(self, driver: Optional[TimelapseDriver] = None) -> None:
        self._driver = driver or TimelapseDriver()
        self._state = _SimState()
        self._tick_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def get_status(self) -> SimulationStatus:
        return SimulationStatus(
            state=self._state.state,
            current_sim_time=self._state.current_sim_time,
        )

    async def _tick_loop(self) -> None:
        try:
            while self._state.state != SimulationState.IDLE:
                if self._state.state == SimulationState.RUNNING:
                    await self._driver.run_tick()
                    self._state.current_sim_time = self._driver.current_sim_time
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    async def start(self) -> SimulationStatus:
        async with self._lock:
            self._state.state = SimulationState.RUNNING
            self._state.current_sim_time = self._driver.current_sim_time
            if self._tick_task and not self._tick_task.done():
                self._tick_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._tick_task
            self._tick_task = asyncio.create_task(self._tick_loop())
            return self.get_status()

    async def manual_tick(self) -> dict:
        async with self._lock:
            if self._state.state != SimulationState.RUNNING:
                return {"status": "skipped"}
            await self._driver.run_tick()
            self._state.current_sim_time = self._driver.current_sim_time
            return {
                "status": "ok",
                "sim_time": self._state.current_sim_time,
            }

    async def pause(self) -> dict:
        async with self._lock:
            if self._state.state == SimulationState.RUNNING:
                self._state.state = SimulationState.PAUSED
            return {"status": "ok"}

    async def resume(self) -> dict:
        async with self._lock:
            if self._state.state == SimulationState.PAUSED:
                self._state.state = SimulationState.RUNNING
                if self._tick_task is None or self._tick_task.done():
                    self._tick_task = asyncio.create_task(self._tick_loop())
            return {"status": "ok"}

    async def restart(self) -> dict:
        async with self._lock:
            if self._tick_task and not self._tick_task.done():
                self._tick_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._tick_task
                self._tick_task = None
            self._state = _SimState()
            try:
                self._driver.restart()
            except Exception:
                pass
            return {"status": "ok"}

    async def shutdown(self) -> None:
        async with self._lock:
            if self._tick_task and not self._tick_task.done():
                self._tick_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._tick_task
                self._tick_task = None
            self._state = _SimState()
            try:
                self._driver.restart()
            except Exception:
                pass
