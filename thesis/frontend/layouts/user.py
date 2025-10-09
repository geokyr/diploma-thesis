"""User interface layout components."""

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html

from thesis.common.config import BBOX
from thesis.common.enums import MLTask
from thesis.frontend.utils.formatting import get_ml_task_icon


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


def create_prediction_output(eta_value: str = "-", fuel_value: str = "-", stops_value: str = "-") -> list[dbc.Row]:
    """
    Create the prediction output layout with the given values.

    Args:
        eta_value (str): ETA value to display.
        fuel_value (str): Fuel consumption value to display.
        stops_value (str): Number of stops value to display.

    Returns:
        list[dbc.Row]: List of rows showing prediction labels with values.
    """
    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.I(className=f"bi {get_ml_task_icon(MLTask.ETA)} me-2"),
                        html.Span("Estimated Time of Arrival", className="fw-semibold"),
                    ],
                    width=8,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Span(eta_value, id="prediction-eta", className="text-end d-block"),
                    width=4,
                ),
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.I(className=f"bi {get_ml_task_icon(MLTask.FUEL)} me-2"),
                        html.Span("Fuel Consumption", className="fw-semibold"),
                    ],
                    width=8,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Span(fuel_value, id="prediction-fuel", className="text-end d-block"),
                    width=4,
                ),
            ],
            className="mb-2",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.I(className=f"bi {get_ml_task_icon(MLTask.STOPS)} me-2"),
                        html.Span("Number of Stops", className="fw-semibold"),
                    ],
                    width=8,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.Span(stops_value, id="prediction-stops", className="text-end d-block"),
                    width=4,
                ),
            ],
        ),
    ]
