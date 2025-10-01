"""Admin tab layout components."""

from dash import dcc, html

from thesis.common.enums import DriftState, SimulationState


def create_admin_layout() -> html.Div:
    """
    Create the admin tab layout with simulation controls and ML task monitoring.

    Returns:
        html.Div: Complete admin tab layout.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Simulation"),
                    html.Button("Start", id="button-start", n_clicks=0, disabled=True),
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


def create_ml_task_card(ml_task: str) -> html.Div:
    """
    Create a card for a single ML task with drift state and metrics chart.

    Args:
        ml_task (str): ML task name.

    Returns:
        html.Div: Card component for the ML task.
    """
    return html.Div(
        [
            html.H4(ml_task),
            html.Div(["Drift: ", html.Span(DriftState.STABLE, id={"type": "drift-state", "ml_task": ml_task})]),
            dcc.Graph(id={"type": "mae-chart", "ml_task": ml_task}),
        ]
    )
