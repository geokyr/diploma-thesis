"""Callbacks for notification polling and display."""

import dash_bootstrap_components as dbc
from dash import Input, Output, dash, no_update

from thesis.common.enums import MLTask
from thesis.common.schemas import Notification
from thesis.frontend.layouts.admin import create_alert
from thesis.frontend.utils.api_client import APIClient
from thesis.frontend.utils.constants import EMPTY_NOTIFICATIONS


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
        Input("bootstrap-interval", "n_intervals"),
    )
    def poll_notifications(
        simulation_intervals: int, bootstrap_intervals: int
    ) -> list[dict[str, int | str | MLTask | None]]:
        """
        Poll backend for all notifications.

        Args:
            simulation_intervals (int): Number of simulation intervals triggered.
            bootstrap_intervals (int): Number of bootstrap intervals triggered.

        Returns:
            list[dict[str, int | str | MLTask | None]]: Notifications list.
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
    def update_notification_panel(notifications_data: list[dict[str, int | str | MLTask | None]]) -> list[dbc.Alert]:
        """
        Update the notification panel content.

        Args:
            notifications_data (list[dict[str, int | str | MLTask | None]]): List of notification data.

        Returns:
            list[dbc.Alert]: Panel content.
        """
        try:
            if not notifications_data:
                return EMPTY_NOTIFICATIONS

            notifications = [Notification.model_validate(n) for n in notifications_data]
            notification_items = [create_alert(notification) for notification in reversed(notifications)]

            return notification_items

        except Exception:
            return no_update
