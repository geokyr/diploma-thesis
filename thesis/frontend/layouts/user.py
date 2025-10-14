"""User interface layout components."""

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html

from thesis.common.config import BBOX
from thesis.common.enums import MLTask
from thesis.frontend.utils.formatting import get_ml_task_icon, get_ml_task_title


def create_user_layout() -> dbc.Row:
    """
    Create the user interface layout with interactive map and prediction controls.

    Returns:
        dbc.Row: Complete user interface layout.
    """
    bbox_bounds = [[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]]
    center = [(BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2]

    return dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H5(
                                    [html.I(className="bi bi-pin-map-fill me-2"), "Trip Selection"],
                                    className="text-center mb-0",
                                )
                            ),
                            dbc.CardBody(
                                [
                                    html.P(
                                        "Click on the map to select source and destination",
                                        className="small text-muted text-center mb-2",
                                    ),
                                    html.H6("Source", className="text-center mb-2"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Latitude", className="small text-muted text-center d-block"
                                                    ),
                                                    dbc.Badge(
                                                        "-",
                                                        id="source-latitude",
                                                        color="light",
                                                        className="fs-6 text-dark",
                                                        pill=True,
                                                    ),
                                                ],
                                                width=6,
                                                className="text-center",
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Longitude", className="small text-muted text-center d-block"
                                                    ),
                                                    dbc.Badge(
                                                        "-",
                                                        id="source-longitude",
                                                        color="light",
                                                        className="fs-6 text-dark",
                                                        pill=True,
                                                    ),
                                                ],
                                                width=6,
                                                className="text-center",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.H6("Destination", className="text-center mb-2"),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Latitude", className="small text-muted text-center d-block"
                                                    ),
                                                    dbc.Badge(
                                                        "-",
                                                        id="destination-latitude",
                                                        color="light",
                                                        className="fs-6 text-dark",
                                                        pill=True,
                                                    ),
                                                ],
                                                width=6,
                                                className="text-center",
                                            ),
                                            dbc.Col(
                                                [
                                                    html.Label(
                                                        "Longitude", className="small text-muted text-center d-block"
                                                    ),
                                                    dbc.Badge(
                                                        "-",
                                                        id="destination-longitude",
                                                        color="light",
                                                        className="fs-6 text-dark",
                                                        pill=True,
                                                    ),
                                                ],
                                                width=6,
                                                className="text-center",
                                            ),
                                        ],
                                        className="mb-2",
                                    ),
                                    html.Hr(className="mt-2 mb-2"),
                                    html.Div(
                                        [
                                            dbc.Button(
                                                [html.I(className="bi bi-trash-fill me-2"), "Clear"],
                                                id="button-clear",
                                                color="secondary",
                                                className="fw-semibold me-2",
                                                n_clicks=0,
                                            ),
                                            dbc.Button(
                                                [html.I(className="bi bi-send-fill me-2"), "Predict"],
                                                id="button-predict",
                                                color="primary",
                                                className="fw-semibold",
                                                n_clicks=0,
                                                disabled=True,
                                            ),
                                        ],
                                        className="text-center",
                                    ),
                                ]
                            ),
                        ],
                        className="mb-2",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                html.H5(
                                    [html.I(className="bi bi-bullseye me-2"), "Trip Predictions"],
                                    className="text-center mb-0",
                                )
                            ),
                            dbc.CardBody(
                                id="prediction-output",
                                children=create_prediction_output(),
                            ),
                        ],
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    dl.Map(
                                        id="user-map",
                                        center=center,
                                        bounds=bbox_bounds,
                                        maxBounds=bbox_bounds,
                                        maxBoundsViscosity=1.0,
                                        minZoom=16,
                                        maxZoom=18,
                                        children=[
                                            dl.TileLayer(),
                                            dl.LayerGroup(id="user-markers"),
                                        ],
                                        style={
                                            "width": "100%",
                                            "aspectRatio": "1.601",
                                            "maxWidth": "1281px",
                                        },
                                    ),
                                ],
                                className="p-0",
                            )
                        ],
                        style={"maxWidth": "1283px"},
                    )
                ],
                width=9,
            ),
        ],
    )


def create_prediction_output(
    ml_tasks: list[str] | None = None, predictions: dict[str, str] | None = None
) -> list[dbc.Row]:
    """
    Create the prediction output layout dynamically based on available ML tasks.

    Args:
        ml_tasks (list[str] | None): List of available ML task strings.
        predictions (dict[str, str] | None): Dictionary mapping ML task strings to prediction values.

    Returns:
        list[dbc.Row]: List of rows showing prediction labels with values.
    """
    if ml_tasks is None:
        ml_tasks = [MLTask.ETA.value, MLTask.FUEL.value, MLTask.STOPS.value]

    if predictions is None:
        predictions = {}

    rows = []
    for i, ml_task_str in enumerate(ml_tasks):
        ml_task = MLTask(ml_task_str)
        value = predictions[ml_task_str] if ml_task_str in predictions else "-"

        row = dbc.Row(
            [
                dbc.Col(
                    [
                        html.I(className=f"bi {get_ml_task_icon(ml_task)} me-2"),
                        html.Span(get_ml_task_title(ml_task), className="fw-semibold"),
                    ],
                    width=8,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Span(value, id=f"prediction-{ml_task_str}", className="text-end d-block"),
                    width=4,
                ),
            ],
            className="mb-2" if i < len(ml_tasks) - 1 else "",
        )
        rows.append(row)

    return rows
