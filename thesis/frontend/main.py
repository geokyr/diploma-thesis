import atexit

import dash
import plotly.graph_objs as go
from dash import Dash, State, ctx, dcc, html, no_update
from dash.dependencies import MATCH, Input, Output

from thesis.common.config import INTERVAL_SECONDS
from thesis.common.enums import DriftState, MLTask, SimulationState
from thesis.common.logger import setup_logger
from thesis.common.schemas import DriftInfo, MetricsResponse, SimulationSnapshot
from thesis.common.service import PlatformServiceConfig
from thesis.frontend.utils.api_client import ApiClient

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)
client = ApiClient(config.backend_url)

# TODO: move callbacks to different file
# TODO: start button should be disabled if no ml tasks are available

app: Dash = dash.Dash("Platform Frontend", suppress_callback_exceptions=True)
app.layout: html.Div = html.Div(
    [
        dcc.Interval(id="bootstrap-interval", interval=1000, n_intervals=0, max_intervals=1),
        dcc.Interval(id="simulation-interval", interval=INTERVAL_SECONDS * 1000, n_intervals=0, disabled=True),
        dcc.Store(
            id="snapshot-store",
            data=SimulationSnapshot(state=SimulationState.IDLE, clock=0, drift_info={}).model_dump(mode="json"),
        ),
        dcc.Store(id="ml-tasks-store", data=[]),
        dcc.Store(id="event-store", data="noop"),
        html.Div(
            [
                html.H2("Simulation"),
                html.Button("Start", id="button-start", n_clicks=0),
                html.Button("Pause", id="button-toggle", n_clicks=0, disabled=True),
                html.Button("Reset", id="button-reset", n_clicks=0, disabled=True),
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
    ]
)


@app.callback(
    Output("event-store", "data"),
    Input("button-start", "n_clicks"),
    Input("button-toggle", "n_clicks"),
    Input("button-reset", "n_clicks"),
    prevent_initial_call=True,
)
def update_event(n_start: int, n_toggle: int, n_reset: int) -> str:
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
def tick_and_apply(
    n_intervals: int, event: str, snapshot_data: dict[str, SimulationState | int | dict[MLTask, DriftInfo]]
) -> tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], bool, int, str]:
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
    Output("ml-cards", "children"),
    Input("ml-tasks-store", "data"),
)
def render_task_cards(ml_tasks: list[str]) -> list:
    try:
        cards = []
        for ml_task in ml_tasks:
            cards.append(
                html.Div(
                    [
                        html.H4(ml_task),
                        html.Div(
                            ["Drift: ", html.Span(DriftState.STABLE, id={"type": "drift-state", "ml_task": ml_task})]
                        ),
                        dcc.Graph(id={"type": "mae-chart", "ml_task": ml_task}),
                    ]
                )
            )

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
    try:
        ml_task = component_id["ml_task"]
        drift_info = snapshot_data["drift_info"]

        if ml_task in drift_info:
            state = drift_info[ml_task].state
            return state

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


@app.callback(
    Output("snapshot-store", "data", allow_duplicate=True),
    Output("ml-tasks-store", "data"),
    Input("bootstrap-interval", "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def bootstrap_snapshot(
    n_intervals: int,
) -> tuple[dict[str, SimulationState | int | dict[MLTask, DriftInfo]], list[str]]:
    try:
        simulation_snapshot = client.simulation_snapshot()
        data = simulation_snapshot.model_dump(mode="json")

        drift_info = data["drift_info"]
        available_ml_tasks = set(drift_info.keys())
        order_ml_tasks = [MLTask.ETA.value, MLTask.FUEL.value, MLTask.STOPS.value]
        ml_tasks = [ml_task for ml_task in order_ml_tasks if ml_task in available_ml_tasks]

        return data, ml_tasks
    except Exception:
        return no_update, no_update


@app.callback(
    Output("button-start", "disabled"),
    Output("button-toggle", "children"),
    Output("button-toggle", "disabled"),
    Output("button-reset", "disabled"),
    Input("snapshot-store", "data"),
)
def update_buttons(data: dict[str, SimulationState | int]) -> tuple[bool, str, bool, bool]:
    try:
        snapshot = SimulationSnapshot.model_validate(data)
        is_idle = snapshot.state == SimulationState.IDLE
        is_running = snapshot.state == SimulationState.RUNNING

        start_disabled = not is_idle
        toggle_label = "Pause" if is_running else "Resume"
        toggle_disabled = is_idle
        reset_disabled = is_idle

        return start_disabled, toggle_label, toggle_disabled, reset_disabled

    except Exception:
        return no_update, no_update, no_update, no_update


@app.callback(
    Output("simulation-state", "children"),
    Output("simulation-clock", "children"),
    Input("snapshot-store", "data"),
)
def update_snapshot(data: dict[str, SimulationState | int]) -> tuple[SimulationState, int]:
    try:
        snapshot = SimulationSnapshot.model_validate(data)
        return snapshot.state, snapshot.clock

    except Exception:
        return no_update, no_update


@atexit.register
def shutdown_client() -> None:
    try:
        client.clear()
    except Exception:
        return


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
