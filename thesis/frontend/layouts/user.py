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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        html.H5(
                                            [html.I(className="bi bi-pin-map-fill me-2"), "Trip Selection"],
                                            className="mb-0 text-center",
                                        )
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                "Click on the map to select source and destination",
                                                className="text-muted mb-3 small text-center",
                                            ),
                                            html.H6("Source", className="text-center mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Latitude",
                                                                className="small text-muted d-block text-center",
                                                            ),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="source-latitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                    pill=True,
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Longitude",
                                                                className="small text-muted d-block text-center",
                                                            ),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="source-longitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                    pill=True,
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.H6("Destination", className="text-center mb-2"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Latitude",
                                                                className="small text-muted d-block text-center",
                                                            ),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="destination-latitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                    pill=True,
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Label(
                                                                "Longitude",
                                                                className="small text-muted d-block text-center",
                                                            ),
                                                            html.Div(
                                                                dbc.Badge(
                                                                    "-",
                                                                    id="destination-longitude",
                                                                    color="light",
                                                                    className="text-dark fs-6",
                                                                    pill=True,
                                                                ),
                                                                className="text-center",
                                                            ),
                                                        ],
                                                        width=6,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Hr(className="my-3"),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        html.Span(
                                                            [html.I(className="bi bi-trash-fill me-1"), "Clear"],
                                                            className="d-flex align-items-center justify-content-center",
                                                        ),
                                                        id="button-clear",
                                                        color="secondary",
                                                        className="fw-semibold me-2",
                                                        n_clicks=0,
                                                    ),
                                                    dbc.Button(
                                                        html.Span(
                                                            [html.I(className="bi bi-send-fill me-1"), "Predict"],
                                                            className="d-flex align-items-center justify-content-center",
                                                        ),
                                                        id="button-predict",
                                                        color="primary",
                                                        className="fw-semibold",
                                                        n_clicks=0,
                                                        disabled=True,
                                                    ),
                                                ],
                                                className="d-flex justify-content-center",
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
                                            className="mb-0 text-center",
                                        )
                                    ),
                                    dbc.CardBody(
                                        id="prediction-output",
                                        children=[
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.I(className="bi bi-clock-fill me-2"),
                                                            html.Span(
                                                                "Estimated Time of Arrival", className="fw-semibold"
                                                            ),
                                                        ],
                                                        width=8,
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "-", id="prediction-eta", className="text-end d-block"
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.I(className="bi bi-fuel-pump-fill me-2"),
                                                            html.Span("Fuel Consumption", className="fw-semibold"),
                                                        ],
                                                        width=8,
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "-", id="prediction-fuel", className="text-end d-block"
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                                className="mb-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.I(className="bi bi-stoplights-fill me-2"),
                                                            html.Span("Number of Stops", className="fw-semibold"),
                                                        ],
                                                        width=8,
                                                        className="d-flex align-items-center",
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "-", id="prediction-stops", className="text-end d-block"
                                                        ),
                                                        width=4,
                                                    ),
                                                ],
                                            ),
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
