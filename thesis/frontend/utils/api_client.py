"""Frontend API client."""

import httpx

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.enums import MLTask
from thesis.common.schemas import (
    MetricsRequest,
    MetricsResponse,
    NotificationFeed,
    PredictionSingleRequest,
    SimulationSnapshot,
    TripPredictionResponse,
)


class APIClient:
    """Frontend API client."""

    def __init__(self, backend_url: str) -> None:
        self._backend_url = backend_url.rstrip("/")
        self._client = httpx.Client(base_url=self._backend_url, timeout=HTTP_CLIENT_TIMEOUT_SECONDS)

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

    def simulation_metrics(self, ml_task: MLTask) -> MetricsResponse:
        """
        Get the metrics of the simulation for a given ML task.

        Args:
            ml_task (MLTask): ML task to fetch metrics for.

        Returns:
            MetricsResponse: The metrics of the simulation.
        """
        params = MetricsRequest(ml_task=ml_task).model_dump(mode="json")
        response = self._client.get("/simulation/metrics", params=params)
        response.raise_for_status()
        return MetricsResponse.model_validate(response.json())

    def simulation_notifications(self) -> NotificationFeed:
        """
        Get all notifications of the simulation.

        Returns:
            NotificationFeed: Feed of all notifications.
        """
        response = self._client.get("/simulation/notifications")
        response.raise_for_status()
        return NotificationFeed.model_validate(response.json())

    def predict_trip(
        self,
        source_latitude: float,
        source_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
        start_timestamp: int,
    ) -> TripPredictionResponse:
        """
        Predict trip metrics for a given trip.

        Args:
            source_latitude (float): Source latitude.
            source_longitude (float): Source longitude.
            destination_latitude (float): Destination latitude.
            destination_longitude (float): Destination longitude.
            start_timestamp (int): Trip start time.

        Returns:
            TripPredictionResponse: Predictions per ML task.
        """
        payload = PredictionSingleRequest(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
            start_timestamp=start_timestamp,
        ).model_dump()

        response = self._client.post("/predict/trip", json=payload)
        response.raise_for_status()
        return TripPredictionResponse.model_validate(response.json())

    def clear(self) -> None:
        """Clear the API client."""
        self._client.close()
