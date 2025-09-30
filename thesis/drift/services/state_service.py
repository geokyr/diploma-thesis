"""Service for managing drift states per ML task."""

import asyncio

from thesis.common.enums import DriftState, MLTask
from thesis.common.schemas import DriftErrorsResponse


class StateService:
    """Service for managing drift states per ML task."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._state: dict[MLTask, DriftErrorsResponse] = {}

        # TODO: remove it after removing the mock
        self._counters: dict[MLTask, int] = {}

    async def get_state(self, ml_task: MLTask) -> DriftErrorsResponse:
        """
        Get the state for the given ML task.

        Args:
            ml_task (MLTask): ML task to get the state for.

        Returns:
            DriftErrorsResponse: State for the given ML task.
        """
        async with self._lock:
            if ml_task not in self._state:
                self._state[ml_task] = DriftErrorsResponse(state=DriftState.STABLE, start_timestamp=0)

            # TODO: remove it after removing the mock
            if ml_task not in self._counters:
                self._counters[ml_task] = 0
            return self._state[ml_task]

    # TODO: remove it after removing the mock
    async def on_errors_event_mock(self, ml_task: MLTask) -> DriftErrorsResponse:
        """Mock drift detection."""
        async with self._lock:
            if ml_task not in self._state:
                self._state[ml_task] = DriftErrorsResponse(state=DriftState.STABLE, start_timestamp=0)
            if ml_task not in self._counters:
                self._counters[ml_task] = 0

            current = self._state[ml_task]
            if current and current.state == DriftState.DRIFTED:
                return current

            self._counters[ml_task] += 1
            threshold_counter = 125
            if self._counters[ml_task] >= threshold_counter:
                drifted = DriftErrorsResponse(state=DriftState.DRIFTED, start_timestamp=threshold_counter * 300)
                self._state[ml_task] = drifted
                return drifted

            return current

    async def clear(self) -> None:
        """Clear the state service."""
        async with self._lock:
            self._state.clear()
