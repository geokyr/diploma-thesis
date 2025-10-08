"""Base layout with stores, intervals, and main structure."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from thesis.common.config import INTERVAL_SECONDS
from thesis.common.enums import SimulationState
from thesis.common.schemas import SimulationSnapshot
from thesis.frontend.layouts.admin import create_admin_layout
from thesis.frontend.layouts.user import create_user_layout


def create_base_layout() -> html.Div:
    """
    Create the root layout with all stores, intervals, tabs, and content.

    Returns:
        html.Div: Complete application layout.
    """
    return html.Div(
        [
            dcc.Interval(id="bootstrap-interval", interval=INTERVAL_SECONDS * 1000, n_intervals=0),
            dcc.Interval(id="simulation-interval", interval=INTERVAL_SECONDS * 1000, n_intervals=0, disabled=True),
            dcc.Store(
                id="snapshot-store",
                data=SimulationSnapshot(state=SimulationState.IDLE, clock=0, drift_info={}).model_dump(mode="json"),
            ),
            dcc.Store(id="ml-tasks-store", data=[]),
            dcc.Store(id="event-store", data="noop"),
            dcc.Store(id="user-source-store", data=None),
            dcc.Store(id="user-destination-store", data=None),
            dcc.Store(id="notifications-store", data=[]),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H1("Platform", className="mt-3 mb-3"),
                                width=12,
                            ),
                        ]
                    ),
                    dbc.Tabs(
                        id="main-tabs",
                        active_tab="admin",
                        className="nav-justified w-100",
                        children=[
                            dbc.Tab(
                                label="Admin Dashboard",
                                tab_id="admin",
                                activeTabClassName="fw-bold",
                                children=[html.Div(create_admin_layout(), className="mt-3")],
                            ),
                            dbc.Tab(
                                id="user-interface-tab",
                                label="User Interface",
                                tab_id="user",
                                activeTabClassName="fw-bold",
                                disabled=True,
                                children=[html.Div(create_user_layout(), className="mt-3")],
                            ),
                        ],
                    ),
                    html.Div(id="user-interface-tab-tooltip-container"),
                ],
                fluid=True,
            ),
        ]
    )
