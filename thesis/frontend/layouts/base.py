"""Base layout with stores, intervals, and main structure."""

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
            html.Header(
                [
                    html.H1("Platform"),
                    dcc.Tabs(
                        id="main-tabs",
                        value="admin",
                        children=[
                            dcc.Tab(label="Admin Tab", value="admin", children=[create_admin_layout()]),
                            dcc.Tab(
                                id="user-tab",
                                label="User Tab",
                                value="user",
                                disabled=True,
                                children=[create_user_layout()],
                            ),
                        ],
                    ),
                    html.Div(id="user-tab-tooltip-container"),
                ]
            ),
        ]
    )
