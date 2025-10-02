"""Callbacks for user tab interactions and predictions."""

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import Input, Output, State, ctx, dash, html, no_update

from thesis.common.config import BBOX
from thesis.common.enums import MLTask, SimulationState
from thesis.common.schemas import DriftInfo, SimulationSnapshot
from thesis.frontend.utils.api_client import APIClient


def register_user_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all user tab callbacks for map interactions and predictions.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("user-tab", "disabled"),
        Output("user-tab-tooltip-container", "children"),
        Input("snapshot-store", "data"),
    )
    def toggle_user_tab_and_tooltip(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
    ) -> tuple[bool, dbc.Tooltip | None]:
        """
        Enable or disable user tab and relevant tooltip based on simulation state.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.

        Returns:
            tuple[bool, dbc.Tooltip | None]: Tab disabled state and tooltip component.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)
            is_paused = snapshot.state == SimulationState.PAUSED

            tab_disabled = not is_paused

            tooltip = (
                dbc.Tooltip(
                    "Start a simulation and pause it to access the User Tab", target="user-tab", placement="top"
                )
                if is_paused
                else None
            )

            return tab_disabled, tooltip

        except Exception:
            return no_update, no_update

    @app.callback(
        Output("user-source-store", "data"),
        Output("user-destination-store", "data"),
        Input("user-map", "clickData"),
        Input("button-clear", "n_clicks"),
        State("user-source-store", "data"),
        State("user-destination-store", "data"),
        State("snapshot-store", "data"),
        prevent_initial_call=True,
    )
    def handle_user_map_click(
        click_data: dict[str, dict[str, float]] | None,
        n_clear: int,
        source_data: dict[str, float] | None,
        destination_data: dict[str, float] | None,
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
    ) -> tuple[dict[str, float] | None, dict[str, float] | None]:
        """
        Handle map clicks to select source and destination points.

        Args:
            click_data (dict[str, dict[str, float]] | None): Click event data containing latlng coordinates.
            n_clear (int): Number of clear button clicks.
            source_data (dict[str, float] | None): Current source point data.
            destination_data (dict[str, float] | None): Current destination point data.
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.

        Returns:
            tuple[dict[str, float] | None, dict[str, float] | None]: Updated source and destination data.
        """
        try:
            trigger = ctx.triggered_id
            if trigger == "button-clear":
                return None, None

            if not click_data or "latlng" not in click_data:
                return no_update, no_update

            latlng = click_data["latlng"]
            if "lat" not in latlng or "lng" not in latlng:
                return no_update, no_update

            latitude, longitude = float(latlng["lat"]), float(latlng["lng"])

            minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude = BBOX
            if not (
                minimum_latitude <= latitude <= maximum_latitude and minimum_longitude <= longitude <= maximum_longitude
            ):
                return no_update, no_update

            point = {"longitude": longitude, "latitude": latitude}

            if source_data is None:
                return point, destination_data
            if destination_data is None:
                return source_data, point

            return point, None

        except Exception:
            return no_update, no_update

    @app.callback(
        Output("source-latitude", "children"),
        Output("source-longitude", "children"),
        Output("destination-latitude", "children"),
        Output("destination-longitude", "children"),
        Input("user-source-store", "data"),
        Input("user-destination-store", "data"),
    )
    def populate_coordinate_fields(
        source_data: dict[str, float] | None, destination_data: dict[str, float] | None
    ) -> tuple[str, str, str, str]:
        """
        Populate coordinate display fields with selected points.

        Args:
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.

        Returns:
            tuple[str, str, str, str]: Formatted latitude and longitude strings for source and destination.
        """
        try:
            source_latitude = f"{source_data['latitude']:.6f}" if source_data else "-"
            source_longitude = f"{source_data['longitude']:.6f}" if source_data else "-"
            destination_latitude = f"{destination_data['latitude']:.6f}" if destination_data else "-"
            destination_longitude = f"{destination_data['longitude']:.6f}" if destination_data else "-"

            return source_latitude, source_longitude, destination_latitude, destination_longitude

        except Exception:
            return no_update, no_update, no_update, no_update

    @app.callback(
        Output("user-markers", "children"),
        Input("user-source-store", "data"),
        Input("user-destination-store", "data"),
    )
    def update_map_markers(
        source_data: dict[str, float] | None, destination_data: dict[str, float] | None
    ) -> list[dl.CircleMarker | dl.Polyline]:
        """
        Update map markers for source and destination points.

        Args:
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.

        Returns:
            list[dl.CircleMarker | dl.Polyline]: List of map marker components.
        """
        try:
            children = []
            if source_data is not None:
                children.append(
                    dl.Marker(
                        position=[source_data["latitude"], source_data["longitude"]],
                        children=[dl.Tooltip("Source")],
                    )
                )
            if destination_data is not None:
                children.append(
                    dl.Marker(
                        position=[destination_data["latitude"], destination_data["longitude"]],
                        children=[dl.Tooltip("Destination")],
                    )
                )
            if source_data is not None and destination_data is not None:
                children.append(
                    dl.Polyline(
                        positions=[
                            [source_data["latitude"], source_data["longitude"]],
                            [destination_data["latitude"], destination_data["longitude"]],
                        ],
                        dashArray="15, 15",
                    )
                )
            return children
        except Exception:
            return no_update

    @app.callback(
        Output("button-predict", "disabled"),
        Input("snapshot-store", "data"),
        Input("user-source-store", "data"),
        Input("user-destination-store", "data"),
    )
    def toggle_predict_button(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
        source_data: dict[str, float] | None,
        destination_data: dict[str, float] | None,
    ) -> bool:
        """
        Enable or disable predict button based on simulation state and point selection.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data from store.
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.

        Returns:
            bool: Predict button disabled state.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)
            is_paused = snapshot.state == SimulationState.PAUSED
            both_selected = bool(source_data) and bool(destination_data)

            return not (is_paused and both_selected)

        except Exception:
            return no_update

    @app.callback(
        Output("prediction-output", "children"),
        Input("button-predict", "n_clicks"),
        State("snapshot-store", "data"),
        State("ml-tasks-store", "data"),
        State("user-source-store", "data"),
        State("user-destination-store", "data"),
        prevent_initial_call=True,
    )
    def run_prediction(
        n_predict: int,
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
        ml_tasks: list[str],
        source_data: dict[str, float] | None,
        destination_data: dict[str, float] | None,
    ) -> html.Div:
        """
        Execute prediction request when Predict button is clicked.

        Args:
            n_predict (int): Number of predict button clicks.
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            ml_tasks (list[str]): Available ML tasks.
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.

        Returns:
            html.Div: Prediction results.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)

            details = [
                html.H4("Trip Information"),
                html.Div(
                    [
                        html.Strong("Source: "),
                        f"({source_data['latitude']:.6f}, {source_data['longitude']:.6f})",
                    ]
                ),
                html.Div(
                    [
                        html.Strong("Destination: "),
                        f"({destination_data['latitude']:.6f}, {destination_data['longitude']:.6f})",
                    ]
                ),
                html.Div([html.Strong("Clock: "), f"{snapshot.clock} seconds"]),
                html.Hr(),
            ]

            has_eta = MLTask.ETA.value in ml_tasks
            if has_eta:
                details.append(html.Div("ETA prediction endpoint will be integrated in the next step. "))

            return html.Div(details)

        except Exception:
            return no_update
