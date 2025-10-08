"""Admin dashboard layout components."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from thesis.common.enums import DriftState, SimulationState
from thesis.common.schemas import Notification
from thesis.frontend.utils.format import format_ml_task_title, format_notification_header, format_simulation_timestamp


def create_admin_layout() -> html.Div:
    """
    Create the admin dashboard layout with simulation controls and ML task monitoring.

    Returns:
        html.Div: Complete admin dashboard layout.
    """
    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H4("Simulation Controls"),
                                    dbc.Button("Start", id="button-start", n_clicks=0, disabled=True),
                                    dbc.Button("Pause", id="button-toggle", n_clicks=0, disabled=True),
                                    dbc.Button("Reset", id="button-reset", n_clicks=0, disabled=True),
                                    # TODO: get default clock from store
                                    html.Div(
                                        [
                                            html.Strong("Clock: "),
                                            html.Span(format_simulation_timestamp(0), id="simulation-clock"),
                                        ]
                                    ),
                                    # TODO: get default state from store
                                    html.Div(
                                        [
                                            html.Strong("Status: "),
                                            html.Span(SimulationState.IDLE, id="simulation-state"),
                                        ]
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    html.H4("Model Status"),
                                    html.Div(id="ml-status-list", children=[html.Div("No predictors available")]),
                                ],
                                width=6,
                            ),
                        ],
                    ),
                    html.Div(id="ml-cards", children=[html.Div("No predictors available")]),
                ],
                width=9,
            ),
            dbc.Col(
                [
                    html.H3("Notifications", style={"marginBottom": "1rem"}),
                    html.Div(
                        id="notification-panel-content",
                        children=[html.P("No notifications")],
                    ),
                ],
                width=3,
            ),
        ],
    )


def create_ml_task_card(ml_task: str) -> html.Div:
    """
    Create a card for a single ML task with drift state and metrics chart.

    Args:
        ml_task (str): ML task name.

    Returns:
        html.Div: Card component for the ML task.
    """
    title = format_ml_task_title(ml_task)

    return html.Div(
        [
            html.H4(title),
            dcc.Graph(
                id={"type": "mae-chart", "ml_task": ml_task},
                config={"displayModeBar": False},
            ),
        ]
    )


def create_ml_status_item(ml_task: str) -> html.Div:
    """
    Create a status item for a single ML task.

    Args:
        ml_task (str): ML task name.

    Returns:
        html.Div: Status item component for the ML task.
    """
    title = format_ml_task_title(ml_task)

    return html.Div(
        [
            html.Strong(f"{title}: "),
            html.Span(DriftState.STABLE, id={"type": "drift-state", "ml_task": ml_task}),
        ]
    )


def create_alert(notification: Notification) -> dbc.Alert:
    """
    Create a single alert.

    Args:
        notification (Notification): Notification to display.

    Returns:
        dbc.Alert: Alert component.
    """
    header = format_notification_header(notification)

    return dbc.Alert(
        [
            html.Strong(notification.message),
            html.Br(),
            html.Small(header),
        ]
    )
