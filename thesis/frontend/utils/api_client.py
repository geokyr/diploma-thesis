"""Frontend API client."""

import httpx

from thesis.common.config import HTTP_CLIENT_TIMEOUT_SECONDS
from thesis.common.enums import MLTask
from thesis.common.schemas import (
    MetricsRequest,
    MetricsResponse,
    NotificationFeed,
    PredictionSingleRequest,
    RoutePreviewRequest,
    RoutePreviewResponse,
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
        response = self._client.post("/control/start")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_pause(self) -> SimulationSnapshot:
        """
        Pause the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/control/pause")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_resume(self) -> SimulationSnapshot:
        """
        Resume the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/control/resume")
        response.raise_for_status()
        return SimulationSnapshot.model_validate(response.json())

    def simulation_reset(self) -> SimulationSnapshot:
        """
        Reset the simulation.

        Returns:
            SimulationSnapshot: The snapshot of the simulation.
        """
        response = self._client.post("/control/reset")
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

    def preview_route(
        self,
        source_latitude: float,
        source_longitude: float,
        destination_latitude: float,
        destination_longitude: float,
    ) -> RoutePreviewResponse:
        """
        Get route preview polyline for the given source and destination.

        Args:
            source_latitude (float): Source latitude.
            source_longitude (float): Source longitude.
            destination_latitude (float): Destination latitude.
            destination_longitude (float): Destination longitude.

        Returns:
            RoutePreviewResponse: Route polyline as list of (lat, lon) tuples.
        """
        payload = RoutePreviewRequest(
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            destination_latitude=destination_latitude,
            destination_longitude=destination_longitude,
        ).model_dump()

        response = self._client.post("/predict/preview", json=payload)
        response.raise_for_status()
        return RoutePreviewResponse.model_validate(response.json())

    def predict_trip(
        self,
        start_timestamp: int,
        source_x: float,
        source_y: float,
        destination_x: float,
        destination_y: float,
        distance: float,
        edges: list[str],
        minimum_x: float,
        maximum_x: float,
        minimum_y: float,
        maximum_y: float,
    ) -> TripPredictionResponse:
        """
        Predict trip metrics for a given trip.

        Args:
            start_timestamp (int): Trip start time.
            source_x (float): Source x coordinate.
            source_y (float): Source y coordinate.
            destination_x (float): Destination x coordinate.
            destination_y (float): Destination y coordinate.
            distance (float): Trip distance in meters.
            edges (list[str]): List of edge IDs along the trip.
            minimum_x (float): Minimum x coordinate along the route.
            maximum_x (float): Maximum x coordinate along the route.
            minimum_y (float): Minimum y coordinate along the route.
            maximum_y (float): Maximum y coordinate along the route.

        Returns:
            TripPredictionResponse: Predictions per ML task.
        """
        payload = PredictionSingleRequest(
            start_timestamp=start_timestamp,
            source_x=source_x,
            source_y=source_y,
            destination_x=destination_x,
            destination_y=destination_y,
            distance=distance,
            edges=edges,
            minimum_x=minimum_x,
            maximum_x=maximum_x,
            minimum_y=minimum_y,
            maximum_y=maximum_y,
        ).model_dump()

        response = self._client.post("/predict/trip", json=payload)
        response.raise_for_status()
        return TripPredictionResponse.model_validate(response.json())

    def clear(self) -> None:
        """Clear the API client."""
        self._client.close()
