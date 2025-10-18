"""Report store for managing report generation state and content."""

import asyncio

from thesis.common.enums import ReportStatus
from thesis.common.schemas import ReportResponse


class ReportStore:
    """Report store for managing report generation state and content."""

    def __init__(self) -> None:
        self._status: ReportStatus = ReportStatus.NOT_STARTED
        self._content: str | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_report(self) -> ReportResponse:
        """
        Get the current report status and content.

        Returns:
            ReportResponse: Report status and content, if available.
        """
        async with self._lock:
            return ReportResponse(status=self._status, content=self._content)

    async def _set_state(self, status: ReportStatus, content: str | None) -> None:
        """
        Internal method to set state and content atomically.

        Args:
            status (ReportStatus): The status to set.
            content (str | None): The content to set.
        """
        async with self._lock:
            self._status = status
            self._content = content

    async def set_generating(self) -> None:
        """Set the report status to generating."""
        await self._set_state(ReportStatus.GENERATING, None)

    async def set_ready(self, content: str) -> None:
        """
        Set the report as ready with content.

        Args:
            content (str): Generated report content.
        """
        await self._set_state(ReportStatus.READY, content)

    async def set_failed(self) -> None:
        """Set the report generation as failed."""
        await self._set_state(ReportStatus.FAILED, None)

    async def reset(self) -> None:
        """Reset the report store."""
        await self._set_state(ReportStatus.NOT_STARTED, None)

    async def clear(self) -> None:
        """Clear the report store."""
        await self._set_state(ReportStatus.NOT_STARTED, None)
