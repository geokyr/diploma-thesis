"""Callbacks for notification polling and display."""

import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, dash, html, no_update

from thesis.common.enums import MLTask
from thesis.common.schemas import Notification
from thesis.frontend.layouts.admin import create_alert
from thesis.frontend.utils.api_client import APIClient

logger = logging.getLogger(__name__)


def register_notification_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all notification-related callbacks.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("notifications-store", "data"),
        Input("simulation-interval", "n_intervals"),
    )
    def poll_notifications(n_intervals: int) -> list[dict[str, str | int | MLTask | None]]:
        """
        Poll backend for all notifications.

        Args:
            n_intervals (int): Number of intervals triggered.

        Returns:
            list[dict[str, str | int | MLTask | None]]: Notifications list.
        """
        try:
            notification_feed = client.simulation_notifications()
            return [n.model_dump(mode="json") for n in notification_feed.notifications]

        except Exception:
            return no_update

    @app.callback(
        Output("notification-panel-content", "children"),
        Input("notifications-store", "data"),
    )
    def update_notification_panel(notifications_data: list[dict[str, str | int | MLTask | None]]) -> list[dbc.Alert]:
        """
        Update the notification panel content.

        Args:
            notifications_data (list[dict[str, str | int | MLTask | None]]): List of notification data.

        Returns:
            list[dbc.Alert]: Panel content.
        """
        try:
            if not notifications_data:
                return [html.P("No notifications", className="text-muted mb-0")]

            notifications = [Notification.model_validate(n) for n in notifications_data]
            notification_items = [create_alert(notification) for notification in reversed(notifications)]

            return notification_items

        except Exception:
            return no_update
