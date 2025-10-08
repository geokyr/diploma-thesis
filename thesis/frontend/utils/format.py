"""Utility functions for formatting."""

from thesis.common.enums import MLTask
from thesis.common.schemas import Notification

_ML_TASK_TITLES = {
    MLTask.ETA: "Estimated Time of Arrival",
    MLTask.FUEL: "Fuel Consumption",
    MLTask.STOPS: "Number of Stops",
}

_ML_TASK_UNITS = {
    MLTask.ETA: "seconds",
    MLTask.FUEL: "liters",
    MLTask.STOPS: "stops",
}


def format_simulation_timestamp(timestamp: int) -> str:
    """
    Format simulation timestamp.

    Args:
        timestamp (int): Simulation timestamp in seconds.

    Returns:
        str: Formatted timestamp string in "Day X HH:MM" format.
    """
    day = (timestamp // 36000) + 1
    remaining_seconds = timestamp % 36000
    hours = remaining_seconds // 3600
    minutes = (remaining_seconds % 3600) // 60

    return f"Day {day:02d} - {hours:02d}:{minutes:02d}"


def format_ml_task_title(ml_task: MLTask | str) -> str:
    """
    Format ML task to its full title.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Full title of the ML task.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _ML_TASK_TITLES[ml_task]


def format_ml_task_unit(ml_task: MLTask | str) -> str:
    """
    Get the unit for an ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Unit for the ML task.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _ML_TASK_UNITS[ml_task]


def format_notification_header(notification: Notification) -> str:
    """
    Format notification header with ML task, if present, and timestamp.

    Args:
        notification (Notification): Notification to format.

    Returns:
        str: Formatted header string.
    """
    timestamp_str = format_simulation_timestamp(notification.timestamp)

    return f"{timestamp_str} - {notification.ml_task.upper()}" if notification.ml_task else timestamp_str
