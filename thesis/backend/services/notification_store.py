"""Notification store for the frontend notifications."""

import asyncio
from collections import deque

from thesis.common.config import NOTIFICATIONS_MAXLEN
from thesis.common.enums import MLTask, NotificationLevel
from thesis.common.schemas import Notification, NotificationFeed


class NotificationStore:
    """Notification store for the frontend notifications."""

    def __init__(self) -> None:
        self._store: deque[Notification] = deque(maxlen=NOTIFICATIONS_MAXLEN)
        self._lock: asyncio.Lock = asyncio.Lock()

    async def push(self, timestamp: int, message: str, level: NotificationLevel, ml_task: MLTask | None = None) -> None:
        """
        Push a notification to the store.

        Args:
            timestamp (int): Simulation timestamp.
            message (str): Notification message.
            level (NotificationLevel): Notification level.
            ml_task (MLTask | None): ML task, if applicable.
        """
        notification = Notification(
            timestamp=timestamp,
            message=message,
            level=level,
            ml_task=ml_task,
        )
        async with self._lock:
            self._store.append(notification)

    async def get_all(self) -> NotificationFeed:
        """
        Get all notifications from the store.

        Returns:
            NotificationFeed: Feed of all notifications.
        """
        async with self._lock:
            notifications = list(self._store)
        return NotificationFeed(notifications=notifications)

    async def reset(self) -> None:
        """Reset the notification store."""
        async with self._lock:
            self._store.clear()

    async def clear(self) -> None:
        """Clear the notification store."""
        async with self._lock:
            self._store.clear()
