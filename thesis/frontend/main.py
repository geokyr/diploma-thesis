import dash
import plotly.graph_objs as go
from dash import Dash, ctx, dcc, html, no_update
from dash.dependencies import Input, Output

from thesis.common.config import INTERVAL_MS, MAX_INTERVALS
from thesis.common.logger import setup_logger
from thesis.common.service import PlatformServiceConfig
from thesis.frontend.utils.api_client import ApiClient

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)
client = ApiClient(config.backend_url)

app: Dash = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Interval(
            id="interval-component", interval=INTERVAL_MS, n_intervals=0, max_intervals=MAX_INTERVALS, disabled=True
        ),
        dcc.Store(
            id="simulation-store",
            data={
                "state": "idle",
                "current_sim_time": 0.0,
                "progress_percent": 0.0,
                "tick_count": 0,
                "event": "init",
            },
        ),
        html.H2("ETA MAE"),
        html.Div(
            [
                html.Button("Start", id="btn-start", n_clicks=0),
                html.Button("Pause", id="btn-toggle", n_clicks=0, disabled=True),
                html.Button("Reset", id="btn-reset", n_clicks=0, disabled=True),
            ]
        ),
        dcc.Graph(id="eta-mae-chart"),
    ]
)


# TODO: tighten dict types
@app.callback(
    Output("eta-mae-chart", "figure"),
    Input("interval-component", "n_intervals"),
    Input("simulation-store", "data"),
)
def update_chart(n_intervals: int, store_data: dict | None):
    hist = client.fetch_history()
    timestamps = hist.get("eta", {}).get("timestamps", [])
    mae_values = hist.get("eta", {}).get("mae_values", [])

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=timestamps[::-1], y=mae_values[::-1], mode="lines+markers"))
    figure.update_layout(yaxis_title="MAE (s)", xaxis_title="Time (s)")

    return figure


@app.callback(
    Output("simulation-store", "data"),
    Input("btn-start", "n_clicks"),
    Input("btn-toggle", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def control_simulation(n_start: int, n_toggle: int, n_reset: int):
    button_id = ctx.triggered_id

    try:
        if button_id == "btn-start":
            client.simulation_start()
        elif button_id == "btn-toggle":
            status = client.fetch_status() or {}
            state = (status.get("state") or "idle").lower()

            if state == "running":
                client.simulation_pause()
            elif state == "paused":
                client.simulation_resume()
        elif button_id == "btn-reset":
            client.simulation_reset()

        latest = client.fetch_status() or {}
        return {
            "state": latest.get("state"),
            "current_sim_time": latest.get("current_sim_time"),
            "progress_percent": latest.get("progress_percent"),
            "tick_count": latest.get("tick_count"),
            "event": button_id,
        }

    except Exception:
        return no_update


@app.callback(
    Output("interval-component", "disabled"),
    Output("interval-component", "n_intervals"),
    Input("simulation-store", "data"),
    prevent_initial_call=True,
)
def manage_interval(store_data: dict):
    try:
        state = (store_data).get("state")
        event = (store_data).get("event")

        disabled = state != "running"
        if event in ("btn-start", "btn-reset"):
            return disabled, 0

        return disabled, no_update

    except Exception:
        return no_update, no_update


@app.callback(
    Output("btn-start", "disabled"),
    Output("btn-toggle", "children"),
    Output("btn-toggle", "disabled"),
    Output("btn-reset", "disabled"),
    Input("simulation-store", "data"),
    prevent_initial_call=True,
)
def sync_buttons(store_data: dict):
    try:
        state = store_data.get("state")

        is_idle = state == "idle"
        is_running = state == "running"
        start_disabled = not is_idle
        toggle_label = "Pause" if is_running else "Resume"
        toggle_disabled = is_idle
        reset_disabled = is_idle

        return start_disabled, toggle_label, toggle_disabled, reset_disabled

    except Exception:
        return no_update, no_update, no_update, no_update


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
