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
        RETRAINING: Retraining state.
    """

    STABLE = "stable"
    DRIFTED = "drifted"
    RETRAINING = "retraining"


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


class RetrainStatus(StrEnum):
    """
    Retrain statuses.

    Attributes:
        COMPLETED: Completed status.
        FAILED: Failed status.
        RUNNING: Running status.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class DriftDetectorType(StrEnum):
    """
    Drift detector types.

    Attributes:
        ADWIN: Adaptive Windowing detector.
        PAGE_HINKLEY: Page-Hinkley test detector.
        KSWIN: Kolmogorov-Smirnov Windowing detector.
        SPC: Statistical Process Control detector.
    """

    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"
    KSWIN = "kswin"
    SPC = "spc"


class NotificationLevel(StrEnum):
    """
    Notification levels.

    Attributes:
        SUCCESS: Success level.
        WARNING: Warning level.
        DANGER: Danger level.
    """

    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
