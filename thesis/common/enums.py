"""Shared enums for the project."""

from enum import StrEnum


class MLTask(StrEnum):
    """
    ML tasks.

    Attributes:
        ETA: Estimated time of arrival prediction task.
        FUEL: Fuel consumption prediction task.
        STOPS: Number of stops prediction task.
    """

    ETA = "eta"
    FUEL = "fuel"
    STOPS = "stops"


class DriftState(StrEnum):
    """
    Drift states.

    Attributes:
        STABLE: Stable state.
        DRIFTED: Drifted state.
        COLLECTING: Collecting state.
        RETRAINING: Retraining state.
        SWAPPED: Swapped state.
    """

    STABLE = "stable"
    DRIFTED = "drifted"
    COLLECTING = "collecting"
    RETRAINING = "retraining"
    SWAPPED = "swapped"


class SimulationState(StrEnum):
    """
    Simulation states.

    Attributes:
        IDLE: Idle state.
        RUNNING: Running state.
        PAUSED: Paused state.
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


class PlatformServiceStatus(StrEnum):
    """
    Platform service statuses.

    Attributes:
        HEALTHY: Healthy status.
        UNHEALTHY: Unhealthy status.
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
