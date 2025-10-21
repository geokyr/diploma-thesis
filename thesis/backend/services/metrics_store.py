"""Metrics store for the frontend charts."""

from collections import deque

from thesis.common.config import METRICS_MAXLEN
from thesis.common.enums import MLTask
from thesis.common.schemas import MetricPoint, MetricsResponse


class MetricsStore:
    """Metrics store for the frontend charts."""

    def __init__(self) -> None:
        self._store: dict[MLTask, deque[MetricPoint]] = {}
        self._maxlen: int = METRICS_MAXLEN

    async def push(self, ml_task: MLTask, timestamp: int, mae: float, n_samples: int) -> None:
        """
        Push a metric point to the store for a given ML task.

        Args:
            ml_task (MLTask): ML task.
            timestamp (int): Timestamp of the metric.
            mae (float): Mean absolute error.
            n_samples (int): Number of samples in this metric point.
        """
        if ml_task not in self._store:
            self._store[ml_task] = deque(maxlen=self._maxlen)

        metric_point = MetricPoint(timestamp=timestamp, mae=mae, n_samples=n_samples)
        self._store[ml_task].append(metric_point)

    async def get_metrics(self, ml_task: MLTask) -> MetricsResponse:
        """
        Get the metrics from the store for a given ML task.

        Args:
            ml_task (MLTask): ML task.

        Returns:
            MetricsResponse: Metrics response.
        """
        if ml_task not in self._store:
            self._store[ml_task] = deque(maxlen=self._maxlen)

        metric_points = list(self._store[ml_task])
        return MetricsResponse(metric_points=metric_points)

    async def reset(self) -> None:
        """Reset the metrics store."""
        for ml_task in list(self._store.keys()):
            if ml_task in self._store:
                self._store[ml_task].clear()

    async def clear(self) -> None:
        """Clear the metrics store."""
        for ml_task in list(self._store.keys()):
            if ml_task in self._store:
                del self._store[ml_task]
