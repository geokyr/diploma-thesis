"""Frontend API client."""

import httpx

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.enums import MLTask
from thesis.common.schemas import (
    MetricsRequest,
    MetricsResponse,
    SimulationSnapshot,
)


class ApiClient:
    """Frontend API client."""

    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self._backend_url, timeout=HTTP_CLIENT_TIMEOUT_SECONDS)

    async def simulation_start(self) -> SimulationSnapshot:
        """
        Start the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = await self._client.post("/simulation/start")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    async def simulation_pause(self) -> SimulationSnapshot:
        """
        Pause the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = await self._client.post("/simulation/pause")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    async def simulation_resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = await self._client.post("/simulation/resume")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    async def simulation_reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = await self._client.post("/simulation/reset")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    async def simulation_snapshot(self) -> SimulationSnapshot:
        """
        Get the snapshot of the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = await self._client.get("/simulation/snapshot")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    async def simulation_metrics(self, ml_task: MLTask) -> MetricsResponse:
        """
        Get the metrics of the simulation for a given ML task.

        Args:
            ml_task (MLTask): ML task to fetch metrics for.

        Returns:
            MetricsResponse: The metrics of the simulation.
        """
        params = MetricsRequest(ml_task=ml_task).model_dump(mode="json")
        response = await self._client.get("/simulation/metrics", params=params)
        response.raise_for_status()
        return MetricsResponse.model_validate(response.json())

    async def clear(self) -> None:
        """Clear the API client."""
        await self._client.aclose()
