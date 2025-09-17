from collections import deque
from dataclasses import dataclass

from thesis.common.schemas import MetricsResponse


@dataclass(slots=True)
class MetricPoint:
    timestamp: int
    mae: float


class MetricsStore:
    def __init__(self) -> None:
        self._buffer: deque[MetricPoint] = deque()

    def push(self, timestamp: int, mae: float) -> None:
        self._buffer.append(MetricPoint(timestamp=timestamp, mae=mae))

    def get_all(self) -> MetricsResponse:
        metric_points = list(self._buffer)
        timestamps = [point.timestamp for point in metric_points]
        maes = [point.mae for point in metric_points]

        return MetricsResponse(
            timestamps=timestamps,
            maes=maes,
        )

    def clear(self) -> None:
        self._buffer.clear()
