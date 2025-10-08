"""Admin dashboard layout components."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from thesis.common.enums import DriftState, SimulationState
from thesis.common.schemas import Notification
from thesis.frontend.utils.format import format_notification_header


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
                    html.Div(
                        [
                            html.H2("Simulation"),
                            dbc.Button("Start", id="button-start", n_clicks=0, disabled=True),
                            dbc.Button("Pause", id="button-toggle", n_clicks=0, disabled=True),
                            dbc.Button("Reset", id="button-reset", n_clicks=0, disabled=True),
                            html.Div(["Status: ", html.Span(SimulationState.IDLE, id="simulation-state")]),
                            html.Div(["Clock: ", html.Span(0, id="simulation-clock")]),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3("ML Tasks"),
                            html.Div(id="ml-cards", children=[html.Div("No predictors available")]),
                        ]
                    ),
                ],
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            html.H3("Notifications", style={"marginBottom": "1rem"}),
                            html.Div(
                                id="notification-panel-content",
                                children=[html.P("No notifications")],
                            ),
                        ],
                    )
                ],
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
    return html.Div(
        [
            html.H4(ml_task),
            html.Div(["Drift: ", html.Span(DriftState.STABLE, id={"type": "drift-state", "ml_task": ml_task})]),
            dcc.Graph(id={"type": "mae-chart", "ml_task": ml_task}),
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
