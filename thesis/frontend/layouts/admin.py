"""Admin dashboard layout components."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from thesis.common.enums import DriftState, SimulationState
from thesis.common.schemas import Notification
from thesis.frontend.utils.constants import EMPTY_NOTIFICATIONS, EMPTY_PREDICTORS
from thesis.frontend.utils.formatting import (
    format_simulation_timestamp,
    get_ml_task_icon,
    get_ml_task_title,
    get_notification_level_color,
)


def create_admin_layout() -> dbc.Row:
    """
    Create the admin dashboard layout with simulation controls and ML task monitoring.

    Returns:
        dbc.Row: Complete admin dashboard layout.
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
                                                className="text-center mb-0",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                dbc.Button(
                                                    [html.I(className="bi bi-play-fill me-2"), "Start"],
                                                    id="button-start",
                                                    color="success",
                                                    className="fw-semibold me-2",
                                                    n_clicks=0,
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    [html.I(className="bi bi-pause-fill me-2"), "Pause"],
                                                    id="button-toggle",
                                                    color="warning",
                                                    className="fw-semibold me-2",
                                                    n_clicks=0,
                                                    disabled=True,
                                                ),
                                                dbc.Button(
                                                    [html.I(className="bi bi-arrow-counterclockwise me-2"), "Reset"],
                                                    id="button-reset",
                                                    color="danger",
                                                    className="fw-semibold",
                                                    n_clicks=0,
                                                    disabled=True,
                                                ),
                                            ],
                                            className="d-flex align-items-center justify-content-center h-100",
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
                                                className="text-center mb-0",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    [
                                                        html.Small(
                                                            "Time",
                                                            className="text-center text-muted d-block mb-1",
                                                        ),
                                                        html.Div(
                                                            format_simulation_timestamp(0),
                                                            id="simulation-clock",
                                                            className="fs-5 fw-semibold text-center",
                                                        ),
                                                    ],
                                                    className="text-center",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Small(
                                                            "State",
                                                            className="text-center text-muted d-block mb-1",
                                                        ),
                                                        dbc.Badge(
                                                            SimulationState.IDLE.value.capitalize(),
                                                            id="simulation-state",
                                                            color="secondary",
                                                            className="fs-6",
                                                            pill=True,
                                                        ),
                                                    ],
                                                    className="text-center",
                                                ),
                                            ],
                                            className="d-flex align-items-center justify-content-around h-100",
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
                                                className="text-center mb-0",
                                            )
                                        ),
                                        dbc.CardBody(
                                            [
                                                html.Div(
                                                    id="ml-status-list",
                                                    children=EMPTY_PREDICTORS,
                                                    className="d-flex align-items-center justify-content-around h-100",
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
                    html.Div(id="ml-cards", children=EMPTY_PREDICTORS),
                ],
                width=9,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    dbc.Button(
                                        [html.I(className="bi bi-robot me-2"), "AI Summary Report"],
                                        id="ai-report-button",
                                        color="primary",
                                        className="fs-5 w-100 border-0",
                                        disabled=True,
                                    ),
                                    id="ai-report-button-container",
                                ),
                                html.Div(id="ai-report-tooltip-container"),
                            ],
                            className="p-0",
                        ),
                        className="mb-2",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H5(
                                    [html.I(className="bi bi-bell-fill me-2"), "Notifications"],
                                    className="text-center mb-0",
                                ),
                            ),
                            dbc.CardBody(
                                id="notification-panel-content",
                                children=EMPTY_NOTIFICATIONS,
                                className="overflow-y-auto",
                            ),
                        ],
                        style={"maxHeight": "82.75vh"},
                    ),
                ],
                width=3,
            ),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle([html.I(className="bi bi-robot me-2"), "AI Summary Report"])),
                    dbc.ModalBody(id="report-modal-body", className="overflow-y-auto", style={"maxHeight": "60vh"}),
                    dbc.ModalFooter(
                        dbc.Button(
                            [html.I(className="bi bi-download me-2"), "Download PDF"],
                            id="download-pdf-button",
                            color="primary",
                            disabled=True,
                        ),
                    ),
                ],
                id="report-modal",
                size="xl",
                is_open=False,
                scrollable=True,
                centered=True,
            ),
        ],
    )


def create_ml_task_card(ml_task: str) -> dbc.Card:
    """
    Create a card for a Falsele ML task with metrics chart.

    Args:
        ml_task (str): ML task name.

    Returns:
        dbc.Card: Card component for the ML task.
    """
    title = get_ml_task_title(ml_task)
    icon = get_ml_task_icon(ml_task)

    return dbc.Card(
        [
            dbc.CardHeader(
                html.H5(
                    [html.I(className=f"bi {icon} me-2"), title],
                    className="text-center mb-0",
                )
            ),
            dbc.CardBody(
                [
                    dcc.Graph(id={"type": "mae-chart", "ml_task": ml_task}),
                ]
            ),
        ],
        className="mb-2",
        style={"height": "25vh"},
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
            html.Small(title, className="text-center text-muted d-block mb-1"),
            dbc.Badge(
                DriftState.STABLE.value.capitalize(),
                id={"type": "drift-state", "ml_task": ml_task},
                color="success",
                className="fs-6",
                pill=True,
            ),
        ],
        className="text-center",
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

    content = notification.message
    if notification.ml_task:
        task_name = get_ml_task_title(notification.ml_task)
        content = f"[{task_name}] {content}"

    return dbc.Alert(
        [
            html.Strong(content),
            html.Br(),
            html.Small(time, className="text-muted"),
        ],
        color=get_notification_level_color(notification.level),
        className="mb-2",
    )
