from typing import TypedDict

import dash
import plotly.graph_objs as go
from dash import Dash, ctx, dcc, html, no_update
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


class SimulationStore(TypedDict):
    state: SimulationState
    clock: int
    event: str


app: Dash = dash.Dash("Platform Frontend")
app.layout = html.Div(
    [
        dcc.Interval(id="interval-component", interval=INTERVAL_SECONDS * 1000, n_intervals=0, disabled=True),
        dcc.Store(
            id="simulation-store",
            data=SimulationStore(state=SimulationState.IDLE, clock=0, event="init"),
        ),
        html.Div(
            [
                html.H2("Simulation"),
                html.Button("Start", id="button-start", n_clicks=0),
                html.Button("Pause", id="button-toggle", n_clicks=0, disabled=True),
                html.Button("Reset", id="button-reset", n_clicks=0, disabled=True),
                html.Div(["Status: ", html.Span(SimulationState.IDLE, id="simulation-status")]),
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
    Output("eta-mae-chart", "figure"),
    Input("interval-component", "n_intervals"),
    Input("simulation-store", "data"),
)
def update_chart(n_intervals: int, store_data: SimulationStore):
    metrics: MetricsResponse = client.simulation_metrics()
    timestamps = [point.timestamp for point in metrics.metric_points]
    mae_values = [point.mae for point in metrics.metric_points]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps[::-1], y=mae_values[::-1], mode="lines+markers"))
    figure.update_layout(yaxis_title="MAE (s)", xaxis_title="Time (s)")

    return figure


@app.callback(
    Output("simulation-store", "data"),
    Input("button-start", "n_clicks"),
    Input("button-toggle", "n_clicks"),
    Input("button-reset", "n_clicks"),
    prevent_initial_call=True,
)
def control_simulation(n_start: int, n_toggle: int, n_reset: int):
    button_id = ctx.triggered_id

    try:
        if button_id == "button-start":
            client.simulation_start()
        elif button_id == "button-toggle":
            snapshot: SimulationSnapshot = client.simulation_snapshot()
            state = snapshot.state

            if state == SimulationState.RUNNING:
                client.simulation_pause()
            elif state == SimulationState.PAUSED:
                client.simulation_resume()
        elif button_id == "button-reset":
            client.simulation_reset()

        latest: SimulationSnapshot = client.simulation_snapshot()
        return SimulationStore(state=latest.state, clock=latest.clock, event=button_id)

    except Exception:
        return no_update


@app.callback(
    Output("interval-component", "disabled"),
    Output("interval-component", "n_intervals"),
    Input("simulation-store", "data"),
    prevent_initial_call=True,
)
def manage_interval(store_data: SimulationStore):
    try:
        state = store_data.state
        event = store_data.event

        disabled = state != SimulationState.RUNNING
        if event in ("button-start", "button-reset"):
            return disabled, 0

        return disabled, no_update

    except Exception:
        return no_update, no_update


@app.callback(
    Output("button-start", "disabled"),
    Output("button-toggle", "children"),
    Output("button-toggle", "disabled"),
    Output("button-reset", "disabled"),
    Input("simulation-store", "data"),
    prevent_initial_call=True,
)
def sync_buttons(store_data: SimulationStore):
    try:
        state = store_data.state

        is_idle = state == SimulationState.IDLE
        is_running = state == SimulationState.RUNNING
        start_disabled = not is_idle
        toggle_label = "Pause" if is_running else "Resume"
        toggle_disabled = is_idle
        reset_disabled = is_idle

        return start_disabled, toggle_label, toggle_disabled, reset_disabled

    except Exception:
        return no_update, no_update, no_update, no_update


@app.callback(
    Output("simulation-status", "children"),
    Output("simulation-clock", "children"),
    Input("simulation-store", "data"),
)
def update_simulation_details(store_data: SimulationStore):
    try:
        state = store_data.state
        clock = store_data.clock

        return state, clock

    except Exception:
        return no_update, no_update


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
