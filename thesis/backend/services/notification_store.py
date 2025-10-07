"""Notification store for the frontend notifications."""

import asyncio
from collections import deque
from uuid import uuid4

from thesis.common.config import NOTIFICATIONS_MAXLEN
from thesis.common.enums import MLTask
from thesis.common.schemas import Notification


class NotificationStore:
    """Notification store for the frontend notifications."""

    def __init__(self) -> None:
        self._store: deque[Notification] = deque(maxlen=NOTIFICATIONS_MAXLEN)
        self._lock: asyncio.Lock = asyncio.Lock()

    async def push(self, timestamp: int, ml_task: MLTask, message: str) -> None:
        """
        Push a notification to the store for a given ML task.

        Args:
            timestamp (int): Simulation timestamp.
            ml_task (MLTask): ML task.
            message (str): Notification message.
        """
        async with self._lock:
            self._store.append(
                Notification(
                    id=str(uuid4()),
                    timestamp=timestamp,
                    ml_task=ml_task,
                    message=message,
                )
            )

    async def get_all(self) -> list[Notification]:
        """
        Get all notifications from the store.

        Returns:
            list[Notification]: List of all notifications.
        """
        async with self._lock:
            return list(self._store)

    async def reset(self) -> None:
        """Reset the notification store."""
        async with self._lock:
            self._store.clear()

    async def clear(self) -> None:
        """Clear the notification store."""
        async with self._lock:
            self._store.clear()
