"""Callbacks for notification polling and display."""

import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dash, html, no_update

from thesis.common.enums import MLTask
from thesis.common.schemas import Notification
from thesis.frontend.layouts.admin import create_alert, create_toast
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
        Output("toast-notifications-store", "data", allow_duplicate=True),
        Input("simulation-interval", "n_intervals"),
        Input("event-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def poll_notifications(
        n_intervals: int, event: str
    ) -> tuple[list[dict[str, str | int | MLTask | None]], list[str]]:
        """
        Poll backend for all notifications and clear toasted IDs on reset.

        Args:
            n_intervals (int): Number of intervals triggered.
            event (str): Event from event store.

        Returns:
            tuple[list[dict[str, str | int | MLTask | None]], list[str]]: Notifications list and toasted IDs list.
        """
        try:
            if event == "button-reset":
                return [], []

            notification_feed = client.simulation_notifications()
            return [n.model_dump(mode="json") for n in notification_feed.notifications], no_update

        except Exception:
            return no_update, no_update

    @app.callback(
        Output("notification-panel-content", "is_open"),
        Input("notification-panel-toggle", "n_clicks"),
        State("notification-panel-content", "is_open"),
    )
    def toggle_notification_panel(n_clicks: int, is_open: bool) -> bool:
        """
        Toggle the notification panel open/closed.

        Args:
            n_clicks (int): Number of clicks on toggle button.
            is_open (bool): Current panel state.

        Returns:
            bool: New panel state.
        """
        if n_clicks > 0:
            return not is_open
        return is_open

    @app.callback(
        Output("toast-container", "children"),
        Output("toast-notifications-store", "data", allow_duplicate=True),
        Input("notifications-store", "data"),
        State("toast-notifications-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_toasts(
        notifications_data: list[dict[str, str | int | MLTask | None]], toast_notifications_data: list[str]
    ) -> tuple[list[dbc.Toast], list[str]]:
        """
        Update toast notifications for new notifications only.

        Args:
            notifications_data (list[dict[str, str | int | MLTask | None]]): List of notification data.
            toast_notifications_data (list[str]): List of notification IDs that have already been shown as toasts.

        Returns:
            tuple[list[dbc.Toast], list[str]]: List of toast components and updated toast notifications IDs.
        """
        try:
            if not notifications_data:
                return [], []

            toast_notifications_data = toast_notifications_data or []
            new_notifications = [
                Notification.model_validate(n)
                for n in notifications_data
                if n.get("id") not in toast_notifications_data
            ]

            if not new_notifications:
                return no_update, no_update

            toasts_to_show = new_notifications[-5:]
            toasts = [create_toast(notification) for notification in toasts_to_show]

            new_toasted_ids = toast_notifications_data + [n.id for n in new_notifications]

            return toasts, new_toasted_ids

        except Exception:
            return no_update, no_update

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
                return [html.P("No notifications")]

            notifications = [Notification.model_validate(n) for n in notifications_data]
            notification_items = [create_alert(notification) for notification in reversed(notifications)]

            return notification_items

        except Exception:
            return no_update
