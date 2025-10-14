"""Callbacks for map interactions and predictions."""

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import Input, Output, State, ctx, dash, html, no_update

from thesis.common.config import BBOX
from thesis.common.enums import MLTask, SimulationState
from thesis.common.schemas import DriftInfo, SimulationSnapshot
from thesis.frontend.layouts.user import create_prediction_output
from thesis.frontend.utils.api_client import APIClient
from thesis.frontend.utils.formatting import format_prediction_value


def register_map_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all map callbacks for map interactions and predictions.

    Args:
        app (dash.Dash): The Dash app instance
        client (APIClient): API client for backend communication
    """

    @app.callback(
        Output("user-map", "invalidateSize"),
        Input("main-tabs", "active_tab"),
        State("user-map", "invalidateSize"),
        prevent_initial_call=True,
    )
    def invalidate_map_size(active_tab: str, current_invalidate_size: int | None) -> int:
        """
        Trigger map size invalidation when switching to user tab.

        Args:
            active_tab (str): Currently active tab ID.

        Returns:
            int: Number to force invalidateSize trigger.
        """
        return 1 if current_invalidate_size == 0 else 0

    @app.callback(
        Output("user-interface-tab", "disabled"),
        Output("user-interface-tab-tooltip-container", "children"),
        Input("snapshot-store", "data"),
    )
    def toggle_user_interface_tab_and_tooltip(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
    ) -> tuple[bool, dbc.Tooltip | None]:
        """
        Enable or disable user interface tab and relevant tooltip based on simulation state.

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
                    [
                        html.I(className="bi bi-info-circle-fill me-2"),
                        "Start a simulation and pause it to access the User Interface",
                    ],
                    target="user-interface-tab",
                    placement="top",
                )
                if tab_disabled
                else None
            )

            return tab_disabled, tooltip

        except Exception:
            return no_update, no_update

    @app.callback(
        Output("user-source-store", "data"),
        Output("user-destination-store", "data"),
        Output("user-route-features-store", "data"),
        Output("prediction-output", "children"),
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
    ) -> tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float | list[str]] | None, html.P]:
        """
        Handle map clicks to select source and destination points.

        Args:
            click_data (dict[str, dict[str, float]] | None): Click event data containing latlng coordinates.
            n_clear (int): Number of clear button clicks.
            source_data (dict[str, float] | None): Current source point data.
            destination_data (dict[str, float] | None): Current destination point data.
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.

        Returns:
            tuple[dict[str, float] | None, dict[str, float] | None, dict[str, float | list[str]] | None, html.P]: Updated source, destination, route features, and prediction output.
        """
        try:
            trigger = ctx.triggered_id
            if trigger == "button-clear":
                return None, None, None, create_prediction_output()

            if not click_data or "latlng" not in click_data:
                return no_update, no_update, no_update, no_update

            latlng = click_data["latlng"]
            if "lat" not in latlng or "lng" not in latlng:
                return no_update, no_update, no_update, no_update

            latitude, longitude = float(latlng["lat"]), float(latlng["lng"])

            minimum_longitude, minimum_latitude, maximum_longitude, maximum_latitude = BBOX
            if not (
                minimum_latitude <= latitude <= maximum_latitude and minimum_longitude <= longitude <= maximum_longitude
            ):
                return no_update, no_update, no_update, no_update

            point = {"longitude": longitude, "latitude": latitude}

            if source_data is None:
                return point, destination_data, None, no_update
            if destination_data is None:
                return source_data, point, None, no_update

            return point, None, None, no_update

        except Exception:
            return no_update, no_update, no_update, no_update

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
        Output("user-route-features-store", "data", allow_duplicate=True),
        Input("user-source-store", "data"),
        Input("user-destination-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_map_markers(
        source_data: dict[str, float] | None, destination_data: dict[str, float] | None
    ) -> tuple[list[dl.CircleMarker | dl.Polyline], dict[str, float | list[str]] | None]:
        """
        Update map markers for source and destination points and the actual route preview.

        Args:
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.

        Returns:
            tuple[list[dl.CircleMarker | dl.Polyline], dict[str, float | list[str]] | None]: List of map marker components and route features dict.
        """
        try:
            children = []
            route_features = None

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
                try:
                    route_response = client.preview_route(
                        source_latitude=source_data["latitude"],
                        source_longitude=source_data["longitude"],
                        destination_latitude=destination_data["latitude"],
                        destination_longitude=destination_data["longitude"],
                    )

                    children.append(dl.Polyline(positions=route_response.route))

                    route_features = {
                        "source_x": route_response.source_x,
                        "source_y": route_response.source_y,
                        "destination_x": route_response.destination_x,
                        "destination_y": route_response.destination_y,
                        "distance": route_response.distance,
                        "edges": route_response.edges,
                        "minimum_x": route_response.minimum_x,
                        "maximum_x": route_response.maximum_x,
                        "minimum_y": route_response.minimum_y,
                        "maximum_y": route_response.maximum_y,
                    }

                except Exception:
                    children.append(
                        dl.Polyline(
                            positions=[
                                [source_data["latitude"], source_data["longitude"]],
                                [destination_data["latitude"], destination_data["longitude"]],
                            ],
                            dashArray="15, 15",
                        )
                    )
            return children, route_features
        except Exception:
            return no_update, no_update

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
        Output("prediction-output", "children", allow_duplicate=True),
        Input("button-predict", "n_clicks"),
        State("snapshot-store", "data"),
        State("ml-tasks-store", "data"),
        State("user-source-store", "data"),
        State("user-destination-store", "data"),
        State("user-route-features-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def run_prediction(
        n_predict: int,
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
        ml_tasks: list[str],
        source_data: dict[str, float] | None,
        destination_data: dict[str, float] | None,
        route_features: dict[str, float | list[str]] | None,
    ) -> html.Div:
        """
        Execute prediction request when Predict button is clicked.

        Args:
            n_predict (int): Number of predict button clicks.
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            ml_tasks (list[str]): Available ML tasks.
            source_data (dict[str, float] | None): Source point data.
            destination_data (dict[str, float] | None): Destination point data.
            route_features (dict[str, float | list[str]] | None): Pre-computed route features from preview.

        Returns:
            html.Div: Prediction results.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)

            try:
                response = client.predict_trip(
                    start_timestamp=snapshot.clock,
                    source_x=route_features["source_x"],
                    source_y=route_features["source_y"],
                    destination_x=route_features["destination_x"],
                    destination_y=route_features["destination_y"],
                    distance=route_features["distance"],
                    edges=route_features["edges"],
                    minimum_x=route_features["minimum_x"],
                    maximum_x=route_features["maximum_x"],
                    minimum_y=route_features["minimum_y"],
                    maximum_y=route_features["maximum_y"],
                )

                eta_value = "-"
                fuel_value = "-"
                stops_value = "-"

                if response.predictions:
                    if MLTask.ETA.value in response.predictions:
                        pred = response.predictions[MLTask.ETA.value]
                        if pred.prediction is not None:
                            eta_value = format_prediction_value(MLTask.ETA, pred.prediction)

                    if MLTask.FUEL.value in response.predictions:
                        pred = response.predictions[MLTask.FUEL.value]
                        if pred.prediction is not None:
                            fuel_value = format_prediction_value(MLTask.FUEL, pred.prediction)

                    if MLTask.STOPS.value in response.predictions:
                        pred = response.predictions[MLTask.STOPS.value]
                        if pred.prediction is not None:
                            stops_value = format_prediction_value(MLTask.STOPS, pred.prediction)

                return create_prediction_output(eta_value=eta_value, fuel_value=fuel_value, stops_value=stops_value)

            except Exception:
                return no_update

        except Exception:
            return no_update
