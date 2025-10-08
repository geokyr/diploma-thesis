"""Callbacks for simulation control and state management."""

from dash import Input, Output, State, ctx, dash, html, no_update

from thesis.common.enums import MLTask, SimulationState
from thesis.common.schemas import DriftInfo, SimulationSnapshot
from thesis.frontend.utils.api_client import APIClient
from thesis.frontend.utils.format import format_simulation_timestamp


def get_simulation_state_color(sim_state: SimulationState) -> str:
    """
    Get Bootstrap color for a simulation state.

    Args:
        sim_state (SimulationState): Simulation state.

    Returns:
        str: Bootstrap color name.
    """
    color_map = {
        SimulationState.IDLE: "secondary",
        SimulationState.RUNNING: "success",
        SimulationState.PAUSED: "warning",
    }
    return color_map.get(sim_state, "secondary")


def register_simulation_callbacks(app: dash.Dash, client: APIClient) -> None:
    """
    Register all simulation-related callbacks.

    Args:
        app (dash.Dash): The Dash app instance
        client (ApiClient): API client for backend communication
    """

    @app.callback(
        Output("snapshot-store", "data", allow_duplicate=True),
        Output("ml-tasks-store", "data"),
        Output("bootstrap-interval", "disabled"),
        Input("bootstrap-interval", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def bootstrap_snapshot(
        n_intervals: int,
    ) -> tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], list[str], bool]:
        """
        Bootstrap the application by fetching initial state.

        Args:
            n_intervals (int): Number of intervals since the last call.

        Returns:
            tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], list[str], bool]: Snapshot data, list of ML tasks, and flag for bootstrap interval disabling.
        """
        try:
            simulation_snapshot = client.simulation_snapshot()
            data = simulation_snapshot.model_dump(mode="json")

            drift_info = data["drift_info"]
            available_ml_tasks = set(drift_info.keys())
            order_ml_tasks = [MLTask.ETA.value, MLTask.FUEL.value, MLTask.STOPS.value]
            ml_tasks = [ml_task for ml_task in order_ml_tasks if ml_task in available_ml_tasks]
            disable_bootstrap = bool(ml_tasks)

            return data, ml_tasks, disable_bootstrap

        except Exception:
            return no_update, no_update, no_update

    @app.callback(
        Output("event-store", "data"),
        Input("button-start", "n_clicks"),
        Input("button-toggle", "n_clicks"),
        Input("button-reset", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_event(n_start: int, n_toggle: int, n_reset: int) -> str:
        """Capture which button was clicked and store the event.

        Args:
            n_start (int): Number of clicks on the start button.
            n_toggle (int): Number of clicks on the toggle button.
            n_reset (int): Number of clicks on the reset button.

        Returns:
            str: Event triggered.
        """
        event = str(ctx.triggered_id)
        return event

    @app.callback(
        Output("snapshot-store", "data"),
        Output("simulation-interval", "disabled"),
        Output("simulation-interval", "n_intervals"),
        Output("event-store", "data", allow_duplicate=True),
        Input("simulation-interval", "n_intervals"),
        Input("event-store", "data"),
        State("snapshot-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def handle_tick_or_event(
        n_intervals: int, event: str, snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]]
    ) -> tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], bool, int, str]:
        """
        Handle simulation tick or button event.

        Args:
            n_intervals (int): Number of intervals triggered.
            event (str): Event triggered.
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.

        Returns:
            tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], bool, int, str]: Snapshot data, flag for simulation interval disabling, number of intervals, and event that was triggered.
        """
        try:
            last = SimulationSnapshot.model_validate(snapshot_data)

            if event == "button-start":
                client.simulation_start()
            elif event == "button-toggle":
                if last.state == SimulationState.RUNNING:
                    client.simulation_pause()
                elif last.state == SimulationState.PAUSED:
                    client.simulation_resume()
            elif event == "button-reset":
                client.simulation_reset()

            snapshot = client.simulation_snapshot()

            new_snapshot_data = snapshot.model_dump(mode="json")
            disabled = snapshot.state != SimulationState.RUNNING
            n_intervals = 0 if event in ("button-start", "button-reset") else no_update
            new_event = "noop"

            return new_snapshot_data, disabled, n_intervals, new_event

        except Exception:
            return no_update, no_update, no_update, no_update

    @app.callback(
        Output("button-start", "disabled"),
        Output("button-toggle", "children"),
        Output("button-toggle", "disabled"),
        Output("button-reset", "disabled"),
        Input("snapshot-store", "data"),
        Input("ml-tasks-store", "data"),
    )
    def update_buttons(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]], ml_tasks: list[str]
    ) -> tuple[bool, html.Span, bool, bool]:
        """
        Update button states based on simulation state.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.
            ml_tasks (list[str]): List of ML tasks.

        Returns:
            tuple[bool, html.Span, bool, bool]: Flag for start button disabling, children for toggle button, flag for toggle button disabling, flag for reset button disabling.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)
            is_idle = snapshot.state == SimulationState.IDLE
            is_paused = snapshot.state == SimulationState.PAUSED
            no_tasks_available = not ml_tasks

            start_disabled = (not is_idle) or no_tasks_available
            toggle_label = "Resume" if is_paused else "Pause"
            toggle_icon = "bi bi-play-fill me-1" if is_paused else "bi bi-pause-fill me-1"
            toggle_children = html.Span(
                [html.I(className=toggle_icon), toggle_label],
                className="d-flex align-items-center",
            )
            toggle_disabled = is_idle
            reset_disabled = is_idle

            return start_disabled, toggle_children, toggle_disabled, reset_disabled

        except Exception:
            return no_update, no_update, no_update, no_update

    @app.callback(
        Output("simulation-state", "children"),
        Output("simulation-state", "color"),
        Output("simulation-state", "className"),
        Output("simulation-clock", "children"),
        Input("snapshot-store", "data"),
    )
    def update_snapshot(
        snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]],
    ) -> tuple[str, str, str, str]:
        """Update the displayed simulation state, badge color, className, and clock.

        Args:
            snapshot_data (dict[str, SimulationState | int | dict[MLTask, DriftInfo]]): Snapshot data.

        Returns:
            tuple[str, str, str, str]: Simulation state, badge color, className, and formatted clock.
        """
        try:
            snapshot = SimulationSnapshot.model_validate(snapshot_data)
            formatted_clock = format_simulation_timestamp(snapshot.clock)

            state_color = get_simulation_state_color(snapshot.state)
            state_text = snapshot.state.value.capitalize()
            state_text_class = "fs-6 text-dark" if snapshot.state == SimulationState.PAUSED else "fs-6"

            return state_text, state_color, state_text_class, formatted_clock

        except Exception:
            return no_update, no_update, no_update, no_update
