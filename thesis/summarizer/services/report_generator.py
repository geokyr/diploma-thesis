"""Report generator service using OpenRouter API."""

import json

from openai import AsyncOpenAI

from thesis.common.config import (
    ASYNC_CLIENT_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    SUMMARIZER_SYSTEM_PROMPT,
)
from thesis.common.enums import MLTask
from thesis.common.schemas import MetricsResponse, Notification, ReportGenerationResponse


class ReportGenerator:
    """Report generator using OpenRouter API."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            timeout=ASYNC_CLIENT_TIMEOUT_SECONDS,
        )

    async def generate_report(
        self, notifications: list[Notification], metrics: dict[MLTask, MetricsResponse]
    ) -> ReportGenerationResponse:
        """
        Generate a markdown report from notifications and metrics.

        Args:
            notifications (list[Notification]): List of notifications from the simulation.
            metrics (dict[MLTask, MetricsResponse]): Metrics for all ML tasks.

        Returns:
            ReportGenerationResponse: Generated markdown report.
        """
        user_content = self._format_data_for_prompt(notifications, metrics)

        completion = await self._client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Summarize the following simulation data: {user_content}"},
            ],
        )

        return ReportGenerationResponse(content=completion.choices[0].message.content)

    def _format_data_for_prompt(self, notifications: list[Notification], metrics: dict[MLTask, MetricsResponse]) -> str:
        """
        Format notifications and metrics into JSON structure for the LLM.

        Args:
            notifications (list[Notification]): List of notifications.
            metrics (dict[MLTask, MetricsResponse]): Metrics per ML task.

        Returns:
            str: Formatted prompt with JSON data.
        """
        data = {
            "notifications": [notification.model_dump(mode="json") for notification in notifications],
            "metrics": {
                ml_task: ml_task_metrics.model_dump(mode="json") for ml_task, ml_task_metrics in metrics.items()
            },
        }

        return json.dumps(data, indent=4)

    async def clear(self) -> None:
        """Clear the report generator."""
        await self._client.close()
