"""Metrics store for the frontend charts."""

import asyncio
from collections import deque

from thesis.common.schemas import MetricPoint, MetricsResponse


class MetricsStore:
    """Metrics store for the frontend charts."""

    def __init__(self) -> None:
        self._metric_points: deque[MetricPoint] = deque()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def push(self, timestamp: int, mae: float | None) -> None:
        """
        Push a metric point to the store.

        Args:
            timestamp (int): Timestamp of the metric.
            mae (float | None): Mean absolute error.
        """
        async with self._lock:
            self._metric_points.append(MetricPoint(timestamp=timestamp, mae=mae))

    async def get_metrics(self) -> MetricsResponse:
        """
        Get the metrics from the store.

        Returns:
            MetricsResponse: Metrics response.
        """
        async with self._lock:
            metric_points = list(self._metric_points)
            return MetricsResponse(metric_points=metric_points)

    async def reset(self) -> None:
        """Reset the metrics store."""
        async with self._lock:
            self._metric_points.clear()

    async def clear(self) -> None:
        """Clear the metrics store."""
        await self.reset()
