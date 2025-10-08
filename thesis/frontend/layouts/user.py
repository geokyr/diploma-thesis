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
    return html.Div(
        [
            html.Div(
                [
                    html.H3("Trip Prediction"),
                    html.P(
                        "Move around the simulation map and click once to select the source, then again for the destination."
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dl.Map(
                                id="user-map",
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
                        width=9,
                    ),
                    dbc.Col(
                        [
                            html.H4("Source"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [html.Div(["Latitude: ", html.Span("-", id="source-latitude")])],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [html.Div(["Longitude: ", html.Span("-", id="source-longitude")])],
                                        width=6,
                                    ),
                                ],
                            ),
                            html.H4("Destination"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [html.Div(["Latitude: ", html.Span("-", id="destination-latitude")])],
                                        width=6,
                                    ),
                                    dbc.Col(
                                        [html.Div(["Longitude: ", html.Span("-", id="destination-longitude")])],
                                        width=6,
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    html.Button("Clear", id="button-clear", n_clicks=0),
                                    html.Button("Predict", id="button-predict", n_clicks=0, disabled=True),
                                ],
                            ),
                            html.Div(
                                id="prediction-output",
                            ),
                        ],
                        width=3,
                    ),
                ],
            ),
        ]
    )
