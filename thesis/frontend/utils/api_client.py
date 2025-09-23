"""Frontend API client."""

import requests

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.schemas import MetricsResponse, SimulationSnapshot


class ApiClient:
    """Frontend API client."""

    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._timeout = HTTP_CLIENT_TIMEOUT_SECONDS
        self._session = requests.Session()

    def _get(self, path: str, **kwargs) -> requests.Response:
        """
        Get a response from the backend.

        Args:
            path: The path to the backend.
            **kwargs: Additional keyword arguments to pass to the session.get method.

        Returns:
            requests.Response: The response from the backend.
        """
        return self._session.get(f"{self._backend_url}{path}", timeout=self._timeout, **kwargs)

    def _post(self, path: str, **kwargs) -> requests.Response:
        """
        Post a request to the backend.

        Args:
            path: The path to the backend.
            **kwargs: Additional keyword arguments to pass to the session.post method.

        Returns:
            requests.Response: The response from the backend.
        """
        return self._session.post(f"{self._backend_url}{path}", timeout=self._timeout, **kwargs)

    def simulation_start(self) -> SimulationSnapshot:
        """
        Start the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._post("/simulation/start")
        response.raise_for_status()
        return SimulationSnapshot(**response.json())

    def simulation_pause(self) -> SimulationSnapshot:
        """
        Pause the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._post("/simulation/pause")
        response.raise_for_status()
        return SimulationSnapshot(**response.json())

    def simulation_resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._post("/simulation/resume")
        response.raise_for_status()
        return SimulationSnapshot(**response.json())

    def simulation_reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._post("/simulation/reset")
        response.raise_for_status()
        return SimulationSnapshot(**response.json())

    def simulation_snapshot(self) -> SimulationSnapshot:
        """
        Get the snapshot of the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._get("/simulation/snapshot")
        response.raise_for_status()
        return SimulationSnapshot(**response.json())

    def simulation_metrics(self) -> MetricsResponse:
        """
        Get the metrics of the simulation.

        Returns:
            MetricsResponse: The metrics of the simulation.
        """
        response = self._get("/simulation/metrics")
        response.raise_for_status()
        return MetricsResponse(**response.json())
