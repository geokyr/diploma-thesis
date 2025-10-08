"""Callbacks for monitoring ML task performance and drift detection."""

import plotly.graph_objs as go
from dash import Input, Output, State, dash, html, no_update
from dash.dependencies import MATCH

from thesis.common.enums import DriftState, MLTask, SimulationState
from thesis.common.schemas import DriftInfo, MetricsResponse
from thesis.frontend.layouts.admin import create_ml_status_item, create_ml_task_card
from thesis.frontend.utils.api_client import APIClient
from thesis.frontend.utils.format import format_simulation_timestamp, get_ml_task_unit


def get_drift_state_color(drift_state: DriftState) -> str:
    """
    Get Bootstrap color for a drift state.

    Args:
        drift_state (DriftState): Drift state.

    Returns:
        str: Bootstrap color name.
    """
    color_map = {
        DriftState.STABLE: "success",
        DriftState.DRIFTED: "danger",
        DriftState.RETRAINING: "warning",
    }
    return color_map[drift_state]


def register_monitoring_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all monitoring callbacks for ML task performance.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("ml-cards", "children"),
        Input("ml-tasks-store", "data"),
    )
    def render_task_cards(ml_tasks: list[str]) -> list[html.Div]:
        """
        Dynamically render cards for each available ML task.

        Args:
            ml_tasks (list[str]): List of ML tasks.

        Returns:
            list[html.Div]: List of cards for each ML task.
        """
        try:
            if not ml_tasks:
                return [html.Div("No predictors available", className="text-muted mb-0")]

            cards = []
            for ml_task in ml_tasks:
                cards.append(create_ml_task_card(ml_task))

            cards[-1].className = "mb-0"

            return cards

        except Exception:
            return no_update

    @app.callback(
        Output("ml-status-list", "children"),
        Input("ml-tasks-store", "data"),
    )
    def render_status_list(ml_tasks: list[str]) -> list[html.Div]:
        """
        Dynamically render status items for each available ML task.

        Args:
            ml_tasks (list[str]): List of ML tasks.

        Returns:
            list[html.Div]: List of status items for each ML task.
        """
        try:
            if not ml_tasks:
                return [html.Div("No predictors available", className="text-muted mb-0")]

            items = []
            for ml_task in ml_tasks:
                items.append(create_ml_status_item(ml_task))

            return items

        except Exception:
            return no_update

    @app.callback(
        Output({"type": "drift-state", "ml_task": MATCH}, "children"),
        Output({"type": "drift-state", "ml_task": MATCH}, "color"),
        Output({"type": "drift-state", "ml_task": MATCH}, "className"),
        Input("snapshot-store", "data"),
        State({"type": "drift-state", "ml_task": MATCH}, "id"),
    )
    def update_drift_label(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]], component_id: dict[str, str]
    ) -> tuple[str, str, str]:
        """
        Update drift state label, color, and className for a specific ML task.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            component_id (dict[str, str]): Component ID.

        Returns:
            tuple[str, str, str]: Drift state, badge color, and className.
        """
        try:
            ml_task = component_id["ml_task"]
            drift_info = snapshot_data["drift_info"]

            if ml_task in drift_info:
                state = DriftState(drift_info[ml_task]["state"])
                color = get_drift_state_color(state)
                state_text = state.value.capitalize()
                state_text_class = "fs-6 text-dark" if state == DriftState.RETRAINING else "fs-6"

                return state_text, color, state_text_class

            return no_update, no_update, no_update

        except Exception:
            return no_update, no_update, no_update

    @app.callback(
        Output({"type": "mae-chart", "ml_task": MATCH}, "figure"),
        Input("snapshot-store", "data"),
        State({"type": "mae-chart", "ml_task": MATCH}, "id"),
    )
    def update_chart_for_task(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]], component_id: dict[str, str]
    ) -> go.Figure:
        """
        Update MAE chart for a specific ML task.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            component_id (dict[str, str]): Component ID.

        Returns:
            go.Figure: MAE chart.
        """
        try:
            ml_task = MLTask(component_id["ml_task"])
            metrics: MetricsResponse = client.simulation_metrics(ml_task)

            timestamps = [format_simulation_timestamp(point.timestamp) for point in metrics.metric_points]
            mae_values = [point.mae for point in metrics.metric_points]

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=timestamps, y=mae_values, mode="lines+markers"))
            figure.update_layout(
                template="seaborn",
                xaxis_title="Time",
                yaxis_title=f"MAE ({get_ml_task_unit(ml_task)})",
                xaxis=dict(
                    showticklabels=len(timestamps) > 0,
                    tickangle=-15,
                    nticks=10,
                ),
                yaxis=dict(showticklabels=len(mae_values) > 0),
                margin=dict(l=40, r=10, t=10, b=40),
                height=250,
            )

            return figure

        except Exception:
            return no_update
