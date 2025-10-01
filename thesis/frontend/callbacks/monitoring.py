"""Callbacks for monitoring ML task performance and drift detection."""

import plotly.graph_objs as go
from dash import Input, Output, State, html, no_update
from dash.dependencies import MATCH

from thesis.common.enums import DriftState, MLTask, SimulationState
from thesis.common.schemas import DriftInfo, MetricsResponse
from thesis.frontend.layouts.admin import create_ml_task_card
from thesis.frontend.utils.api_client import APIClient


def register_monitoring_callbacks(app, client: APIClient) -> None:
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
                return [html.Div("No predictors available")]

            cards = []
            for ml_task in ml_tasks:
                cards.append(create_ml_task_card(ml_task))

            return cards

        except Exception:
            return no_update

    @app.callback(
        Output({"type": "drift-state", "ml_task": MATCH}, "children"),
        Input("snapshot-store", "data"),
        State({"type": "drift-state", "ml_task": MATCH}, "id"),
    )
    def update_drift_label(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]], component_id: dict[str, str]
    ) -> DriftState:
        """
        Update drift state label for a specific ML task.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            component_id (dict[str, str]): Component ID.

        Returns:
            DriftState: Drift state.
        """
        try:
            ml_task = component_id["ml_task"]
            drift_info = snapshot_data["drift_info"]

            if ml_task in drift_info:
                return drift_info[ml_task]["state"]

            return no_update

        except Exception:
            return no_update

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

            timestamps = [point.timestamp for point in metrics.metric_points]
            mae_values = [point.mae for point in metrics.metric_points]

            figure = go.Figure()
            figure.add_trace(go.Scatter(x=timestamps[::-1], y=mae_values[::-1], mode="lines+markers"))
            figure.update_layout(yaxis_title="MAE (s)", xaxis_title="Time (s)")

            return figure

        except Exception:
            return no_update
