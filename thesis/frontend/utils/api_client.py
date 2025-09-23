"""Frontend API client."""

import httpx

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.schemas import MetricsResponse, SimulationSnapshot


class ApiClient:
    """Frontend API client."""

    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._client = httpx.Client(base_url=backend_url, timeout=HTTP_CLIENT_TIMEOUT_SECONDS)

    def simulation_start(self) -> SimulationSnapshot:
        """
        Start the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/simulation/start")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_pause(self) -> SimulationSnapshot:
        """
        Pause the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/simulation/pause")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/simulation/resume")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/simulation/reset")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_snapshot(self) -> SimulationSnapshot:
        """
        Get the snapshot of the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.get("/simulation/snapshot")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_metrics(self) -> MetricsResponse:
        """
        Get the metrics of the simulation.

        Returns:
            MetricsResponse: The metrics of the simulation.
        """
        response = self._client.get("/simulation/metrics")
        response.raise_for_status()
        return MetricsResponse.model_validate(response.json())
