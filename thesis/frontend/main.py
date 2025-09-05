"""
Dash frontend for platform.
Includes Simulation Control, Simulation Status, Real-time Metrics and Notifications.
"""

from datetime import datetime

import plotly.graph_objs as go
import requests
from dash import Dash, Input, Output, callback, dcc, html

from thesis.common.config import UPDATE_INTERVAL_MS
from thesis.common.logger import setup_logger
from thesis.common.service import PlatformServiceConfig

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)

app = Dash(__name__)
app.title = "Platform - Phase 1"

app.layout = html.Div(
    [
        html.H1("Platform - Phase 1", style={"text-align": "center", "margin-bottom": "10px"}),
        html.Div(
            [
                html.H3("Simulation Control"),
                html.Button("Start", id="start-button", n_clicks=0, style={"margin-right": "10px", "padding": "5px"}),
                html.Button("Pause", id="pause-button", n_clicks=0, style={"margin-right": "10px", "padding": "5px"}),
                html.Div(id="control-feedback", style={"margin-top": "5x"}),
            ],
            style={"margin-bottom": "10px", "padding": "5px", "border": "1px solid #ccc"},
        ),
        html.Div(
            [
                html.H3("Simulation Status"),
                html.Div(id="status-display"),
            ],
            style={"margin-bottom": "10px", "padding": "5px", "border": "1px solid #ccc"},
        ),
        html.Div(
            [
                html.H3("Real-time Metrics"),
                dcc.Graph(id="eta-mae-graph"),
                html.Div(id="metrics-display"),
            ],
            style={"margin-bottom": "10px"},
        ),
        html.Div(
            [
                html.H3("Notifications"),
                html.Div(
                    id="notifications-display",
                    style={"height": "200px", "overflow-y": "scroll", "padding": "5px", "border": "1px solid #ccc"},
                ),
            ]
        ),
        dcc.Interval(id="interval-component", interval=UPDATE_INTERVAL_MS, n_intervals=0),
    ]
)


@callback(
    Output("control-feedback", "children"),
    Input("start-button", "n_clicks"),
    prevent_initial_call=True,
)
def start_simulation(n_clicks):
    if n_clicks == 0:
        return ""

    try:
        response = requests.post(f"{config.backend_url}/start", timeout=10)
        if response.status_code == 200:
            return html.Div("Simulation started", style={"color": "green"})
        else:
            return html.Div(f"Failed to start: {response.text}", style={"color": "red"})
    except Exception as e:
        return html.Div(f"Connection error: {str(e)}", style={"color": "red"})


@callback(
    Output("control-feedback", "children", allow_duplicate=True),
    Input("pause-button", "n_clicks"),
    prevent_initial_call=True,
)
def pause_simulation(n_clicks):
    if n_clicks == 0:
        return ""

    try:
        response = requests.post(f"{config.backend_url}/pause", timeout=10)
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "")
            return html.Div(f"✅ {message}", style={"color": "green"})
        else:
            return html.Div(f"Failed to pause/resume: {response.text}", style={"color": "red"})
    except Exception as e:
        return html.Div(f"Connection error: {str(e)}", style={"color": "red"})


@callback(
    [Output("status-display", "children"), Output("pause-button", "children")],
    Input("interval-component", "n_intervals"),
)
def update_status(n_intervals):
    try:
        response = requests.get(f"{config.backend_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()

            status_div = html.Div(
                [
                    html.P(f"Active: {'Yes' if status['active'] else 'No'}"),
                    html.P(f"Simulation Time: {status['current_time']}s"),
                    html.P(f"Dataset: {status['dataset']}"),
                    html.P(f"Progress: {status['progress_percent']:.1f}%"),
                    html.P(f"Speed: {status['speed_multiplier']}x"),
                ]
            )
            button_text = "Resume" if not status["active"] else "Pause"

            return status_div, button_text
        else:
            error_div = html.Div("Failed to get status", style={"color": "red"})
            return error_div, "Pause"
    except Exception as e:
        error_div = html.Div(f"Backend not available: {str(e)}", style={"color": "red"})
        return error_div, "Pause"


@callback(
    [Output("eta-mae-graph", "figure"), Output("metrics-display", "children")],
    Input("interval-component", "n_intervals"),
)
def update_metrics(n_intervals):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines+markers", name="ETA-MAE", line=dict(color="blue")))

    fig.update_layout(
        title="ETA - Mean Absolute Error (MAE)", xaxis_title="Time", yaxis_title="MAE (seconds)", height=400
    )

    metrics_text = html.Div(
        [
            html.P("Real-time metrics will be displayed here in Phase 1"),
            html.P("WebSocket integration coming in Phase 2"),
            html.P(f"Last update: {datetime.now().strftime('%H:%M:%S')}"),
        ]
    )

    return fig, metrics_text


@callback(
    Output("notifications-display", "children"),
    Input("interval-component", "n_intervals"),
)
def update_notifications(n_intervals):
    return html.Div(
        [
            html.P("Notification system will be integrated in Phase 2"),
            html.P("Drift detection events will appear here"),
            html.P("System ready for Phase 1 testing"),
        ]
    )


if __name__ == "__main__":
    app.run(host=config.host, port=config.port, debug=config.debug)
