"""Backend HTTP client for Drift service API."""

import httpx

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.schemas import DriftErrorsResponse


class DriftClient:
    """Backend HTTP client for Drift service API."""

    def __init__(self, drift_url: str) -> None:
        self._drift_url = drift_url.rstrip("/")
        self._timeout = HTTP_CLIENT_TIMEOUT_SECONDS
        self._client = httpx.AsyncClient(base_url=self._drift_url, timeout=self._timeout)

    async def get_status(self) -> DriftErrorsResponse:
        """
        Get the status of all ML Tasks from drift service.

        Returns:
            DriftErrorsResponse: Status of all ML Tasks.
        """
        response = await self._client.get("/drift/status")
        response.raise_for_status()
        return DriftErrorsResponse.model_validate(response.json())

    async def clear(self) -> None:
        """Clear the drift client."""
        await self._client.aclose()
