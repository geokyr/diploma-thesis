"""Utility functions for formatting."""

from thesis.common.schemas import Notification


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

    return f"Day {day} {hours:02d}:{minutes:02d}"


def format_notification_header(notification: Notification) -> str:
    """
    Format notification header with ML task, if present, and timestamp.

    Args:
        notification (Notification): Notification to format.

    Returns:
        str: Formatted header string.
    """
    timestamp_str = format_simulation_timestamp(notification.timestamp)

    return f"{notification.ml_task.upper()} - {timestamp_str}" if notification.ml_task else timestamp_str
