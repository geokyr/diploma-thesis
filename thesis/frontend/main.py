import dash
import plotly.graph_objs as go
from dash import Dash, ctx, dcc, html, no_update
from dash.dependencies import Input, Output

from thesis.common.service import PlatformServiceConfig
from thesis.frontend.utils.api_client import ApiClient

config = PlatformServiceConfig()
client = ApiClient(config.backend_url)

app: Dash = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=240, disabled=True),
        html.H2("ETA MAE"),
        html.Div(
            [
                html.Button("Start", id="btn-start", n_clicks=0),
                html.Button("Pause", id="btn-toggle", n_clicks=0, disabled=True),
                html.Button("Reset", id="btn-reset", n_clicks=0, disabled=True),
            ]
        ),
        html.Div(id="simulation-status", style={"display": "none"}),
        dcc.Graph(id="eta-mae-chart"),
    ]
)


@app.callback(
    Output("eta-mae-chart", "figure"),
    [
        Input("interval-component", "n_intervals"),
        Input("simulation-status", "children"),
    ],
)
def update_chart(n_intervals: int, status: str):
    hist = client.fetch_history()
    timestamps = hist.get("eta", {}).get("timestamps", [])
    mae_values = hist.get("eta", {}).get("mae_values", [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps[::-1], y=mae_values[::-1], mode="lines+markers"))
    fig.update_layout(yaxis_title="MAE (s)", xaxis_title="Time (s)")

    return fig


@app.callback(
    Output("simulation-status", "children"),
    [
        Input("btn-start", "n_clicks"),
        Input("btn-toggle", "n_clicks"),
        Input("btn-reset", "n_clicks"),
    ],
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

        return button_id

    except Exception:
        return "Error"


@app.callback(
    Output("interval-component", "disabled"),
    Output("interval-component", "n_intervals"),
    Input("simulation-status", "children"),
    prevent_initial_call=True,
)
def manage_interval(button_id: str):
    try:
        status = client.fetch_status() or {}
        state = (status.get("state") or "idle").lower()

        disabled = state != "running"
        if button_id in ("btn-start", "btn-reset"):
            return disabled, 0

        return disabled, no_update

    except Exception:
        return True, no_update


@app.callback(
    Output("btn-start", "disabled"),
    Output("btn-toggle", "children"),
    Output("btn-toggle", "disabled"),
    Output("btn-reset", "disabled"),
    Input("simulation-status", "children"),
    prevent_initial_call=True,
)
def sync_buttons(_button_id: str):
    try:
        status = client.fetch_status() or {}
        state = (status.get("state") or "idle").lower()

        is_idle = state == "idle"
        toggle_label = "Pause" if state == "running" else "Resume"

        return not is_idle, toggle_label, is_idle, is_idle

    except Exception:
        return no_update, no_update, no_update, no_update


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.is_development)
