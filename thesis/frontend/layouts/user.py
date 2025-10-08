"""User interface layout components."""

import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash import html

from thesis.common.config import BBOX


def create_user_layout() -> html.Div:
    """
    Create the user interface layout with interactive map and prediction controls.

    Returns:
        html.Div: Complete user interface layout.
    """
    bbox_bounds = [[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]]
    center = [(BBOX[1] + BBOX[3]) / 2, (BBOX[0] + BBOX[2]) / 2]

    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H3("Trip Prediction", className="mb-2"),
                            html.P(
                                "Click on the map to select the source and destination points to get predictions for a route.",
                                className="text-muted mb-0",
                            ),
                        ]
                    )
                ],
                className="mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Source", className="mb-0")),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Latitude", className="small text-muted"),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="source-latitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Longitude", className="small text-muted"),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="source-longitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Destination", className="mb-0")),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label("Latitude", className="small text-muted"),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="destination-latitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label("Longitude", className="small text-muted"),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="destination-longitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                ),
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.ButtonGroup(
                                [
                                    dbc.Button(
                                        html.Span(
                                            [html.I(className="bi bi-trash-fill me-1"), "Clear"],
                                            className="d-flex align-items-center",
                                        ),
                                        id="button-clear",
                                        color="secondary",
                                        class_name="fw-semibold",
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        html.Span(
                                            [html.I(className="bi bi-send-fill me-1"), "Predict"],
                                            className="d-flex align-items-center",
                                        ),
                                        id="button-predict",
                                        color="primary",
                                        class_name="fw-semibold",
                                        n_clicks=0,
                                        disabled=True,
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(html.H5("Predictions", className="mb-0")),
                                    dbc.CardBody(
                                        id="prediction-output",
                                        children=[
                                            html.P(
                                                "Select source and destination to get predictions",
                                                className="text-muted mb-0",
                                            )
                                        ],
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
                                                    "borderRadius": "0.3rem",
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
            ),
        ]
    )
