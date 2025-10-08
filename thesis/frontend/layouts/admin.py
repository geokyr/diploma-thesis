"""Admin dashboard layout components."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from thesis.common.schemas import Notification
from thesis.frontend.utils.format import format_simulation_timestamp, get_ml_task_title


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
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [html.I(className="bi bi-gear-fill me-2"), "Simulation Controls"],
                                                className="mb-0 text-center",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            html.Span(
                                                                [html.I(className="bi bi-play-fill me-1"), "Start"],
                                                                className="d-flex align-items-center justify-content-center",
                                                            ),
                                                            id="button-start",
                                                            color="success",
                                                            className="fw-semibold me-2",
                                                            n_clicks=0,
                                                            disabled=True,
                                                        ),
                                                        dbc.Button(
                                                            html.Span(
                                                                [html.I(className="bi bi-pause-fill me-1"), "Pause"],
                                                                className="d-flex align-items-center justify-content-center",
                                                            ),
                                                            id="button-toggle",
                                                            color="warning",
                                                            className="fw-semibold me-2",
                                                            n_clicks=0,
                                                            disabled=True,
                                                        ),
                                                        dbc.Button(
                                                            html.Span(
                                                                [
                                                                    html.I(
                                                                        className="bi bi-arrow-counterclockwise me-1"
                                                                    ),
                                                                    "Reset",
                                                                ],
                                                                className="d-flex align-items-center justify-content-center",
                                                            ),
                                                            id="button-reset",
                                                            color="danger",
                                                            className="fw-semibold",
                                                            n_clicks=0,
                                                            disabled=True,
                                                        ),
                                                    ],
                                                    className="d-flex justify-content-center align-items-center h-100",
                                                ),
                                            ],
                                            className="d-flex justify-content-center align-items-center h-100",
                                        ),
                                    ],
                                    className="h-100",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [
                                                    html.I(className="bi bi-clipboard-data-fill me-2"),
                                                    "Simulation Snapshot",
                                                ],
                                                className="mb-0 text-center",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                html.Small(
                                                                    "Time",
                                                                    className="text-muted mb-1 d-block text-center",
                                                                ),
                                                                html.Div(
                                                                    format_simulation_timestamp(0),
                                                                    id="simulation-clock",
                                                                    className="fw-semibold fs-5 text-center",
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Small(
                                                                    "State",
                                                                    className="text-muted d-block mb-1 text-center",
                                                                ),
                                                                html.Div(
                                                                    dbc.Badge(
                                                                        "Idle",
                                                                        id="simulation-state",
                                                                        color="secondary",
                                                                        className="fs-6",
                                                                        pill=True,
                                                                    ),
                                                                    className="text-center",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                    className="d-flex justify-content-around h-100 align-items-center w-100",
                                                ),
                                            ],
                                        ),
                                    ],
                                    className="h-100",
                                ),
                                width=3,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader(
                                            html.H5(
                                                [html.I(className="bi bi-activity me-2"), "Drift State"],
                                                className="mb-0 text-center",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    id="ml-status-list",
                                                    children=[
                                                        html.P("No predictors available", className="text-muted mb-0")
                                                    ],
                                                    className="d-flex justify-content-around h-100",
                                                )
                                            ],
                                        ),
                                    ],
                                    className="h-100",
                                ),
                                width=6,
                            ),
                        ],
                        className="mb-2",
                    ),
                    html.Div(id="ml-cards", children=[html.P("No predictors available", className="text-muted mb-0")]),
                ],
                width=9,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5(
                                [html.I(className="bi bi-bell-fill me-2"), "Notifications"],
                                className="mb-0 text-center",
                            )
                        ),
                        dbc.CardBody(
                            id="notification-panel-content",
                            children=[html.P("No notifications", className="text-muted mb-0")],
                            style={"overflowY": "auto", "height": "100%"},
                        ),
                    ],
                    className="h-100",
                ),
                width=3,
            ),
        ],
    )


def create_ml_task_card(ml_task: str) -> dbc.Card:
    """
    Create a card for a single ML task with metrics chart.

    Args:
        ml_task (str): ML task name.

    Returns:
        dbc.Card: Card component for the ML task.
    """
    from thesis.common.enums import MLTask

    title = get_ml_task_title(ml_task)

    # Map ML tasks to icons
    icon_map = {
        MLTask.ETA.value: "bi-clock-fill",
        MLTask.FUEL.value: "bi-fuel-pump-fill",
        MLTask.STOPS.value: "bi-stoplights-fill",
    }

    icon_class = icon_map.get(ml_task, "bi-graph-up")

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    [html.I(className=f"bi {icon_class} me-2"), title],
                    className="mb-0 text-center",
                )
            ),
            dbc.CardBody(
                [
                    dcc.Graph(id={"type": "mae-chart", "ml_task": ml_task}),
                ]
            ),
        ],
        className="mb-2",
    )


def create_ml_status_item(ml_task: str) -> html.Div:
    """
    Create a status item for a single ML task.

    Args:
        ml_task (str): ML task name.

    Returns:
        html.Div: Status item component for the ML task.
    """
    title = get_ml_task_title(ml_task)

    return html.Div(
        [
            html.Small(title, className="text-muted d-block mb-1 text-center"),
            html.Div(
                dbc.Badge(
                    "Stable",
                    id={"type": "drift-state", "ml_task": ml_task},
                    color="success",
                    className="fs-6",
                    pill=True,
                ),
                className="text-center",
            ),
        ],
    )


def create_alert(notification: Notification) -> dbc.Alert:
    """
    Create a single alert.

    Args:
        notification (Notification): Notification to display.

    Returns:
        dbc.Alert: Alert component.
    """
    time = format_simulation_timestamp(notification.timestamp)

    message = notification.message
    if notification.ml_task:
        task_name = get_ml_task_title(notification.ml_task.value)
        message = f"[{task_name}] {message}"

    return dbc.Alert(
        [
            html.Strong(message),
            html.Br(),
            html.Small(time, className="text-muted"),
        ],
        color=notification.level.value,
        className="mb-2",
    )
