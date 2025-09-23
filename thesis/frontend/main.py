import dash
import plotly.graph_objs as go
from dash import Dash, State, ctx, dcc, html, no_update
from dash.dependencies import Input, Output

from thesis.common.config import INTERVAL_SECONDS
from thesis.common.enums import SimulationState
from thesis.common.logger import setup_logger
from thesis.common.schemas import MetricsResponse, SimulationSnapshot
from thesis.common.service import PlatformServiceConfig
from thesis.frontend.utils.api_client import ApiClient

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)
client = ApiClient(config.backend_url)


app: Dash = dash.Dash("Platform Frontend")
app.layout = html.Div(
    [
        dcc.Interval(id="interval-component", interval=INTERVAL_SECONDS * 1000, n_intervals=0, disabled=True),
        dcc.Store(id="snapshot-store", data=SimulationSnapshot(state=SimulationState.IDLE, clock=0).to_dict()),
        dcc.Store(id="event-store", data={"event": "noop"}),
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
                html.H3("ETA MAE"),
                dcc.Graph(id="eta-mae-chart"),
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
def update_ui_event(n_start: int, n_toggle: int, n_reset: int):
    return {"event": ctx.triggered_id}


@app.callback(
    Output("snapshot-store", "data"),
    Output("interval-component", "disabled"),
    Output("interval-component", "n_intervals"),
    Output("event-store", "data", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    Input("event-store", "data"),
    State("snapshot-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def tick_and_apply(n_intervals: int, event_data: dict[str, str], snapshot_data: dict[str, SimulationState | int]):
    try:
        last = SimulationSnapshot.from_dict(snapshot_data)
        event = event_data.get("event")

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
        new_snapshot_data = snapshot.to_dict()
        disabled = snapshot.state != SimulationState.RUNNING
        n_intervals = 0 if event in ("button-start", "button-reset") else no_update
        new_event_data = {"event": "noop"}

        return new_snapshot_data, disabled, n_intervals, new_event_data

    except Exception:
        return no_update, no_update, no_update, no_update


@app.callback(
    Output("eta-mae-chart", "figure"),
    Input("snapshot-store", "data"),
)
def update_chart(data: dict[str, SimulationState | int]):
    try:
        metrics: MetricsResponse = client.simulation_metrics()
        timestamps = [point.timestamp for point in metrics.metric_points]
        mae_values = [point.mae for point in metrics.metric_points]

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=timestamps[::-1], y=mae_values[::-1], mode="lines+markers"))
        figure.update_layout(yaxis_title="MAE (s)", xaxis_title="Time (s)")

        return figure
    except Exception:
        return no_update


@app.callback(
    Output("button-start", "disabled"),
    Output("button-toggle", "children"),
    Output("button-toggle", "disabled"),
    Output("button-reset", "disabled"),
    Input("snapshot-store", "data"),
)
def update_buttons(data: dict[str, SimulationState | int]):
    try:
        snapshot = SimulationSnapshot.from_dict(data)
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
def update_snapshot(data: dict[str, SimulationState | int]):
    try:
        snapshot = SimulationSnapshot.from_dict(data)
        return snapshot.state, snapshot.clock

    except Exception:
        return no_update, no_update


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
