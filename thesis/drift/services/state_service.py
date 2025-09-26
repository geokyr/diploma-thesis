"""Service for managing drift states per ML task."""

from threading import RLock

from thesis.common.enums import DriftState, MLTask
from thesis.common.schemas import DriftErrorsResponse


# TODO: add async
class StateService:
    """Service for managing drift states per ML task."""

    def __init__(self) -> None:
        self._lock: RLock = RLock()
        self._state: dict[MLTask, DriftErrorsResponse] = {
            MLTask.ETA: DriftErrorsResponse(state=DriftState.STABLE, start_timestamp=0),
        }

    def get_state(self, ml_task: MLTask) -> DriftErrorsResponse:
        """
        Get the state for the given ML task.

        Args:
            ml_task (MLTask): ML task to get the state for.

        Returns:
            DriftErrorsResponse: State for the given ML task.
        """
        with self._lock:
            return self._state[ml_task]

    def set_state(self, ml_task: MLTask, state: DriftState, start_timestamp: int) -> None:
        """
        Set the state for the given ML task.

        Args:
            ml_task (MLTask): ML task to set the state for.
        """
        with self._lock:
            self._state[ml_task] = DriftErrorsResponse(state=state, start_timestamp=start_timestamp)

    def clear(self) -> None:
        """Clear the state service."""
        with self._lock:
            self._state.clear()
