"""User tab layout components."""

import dash_leaflet as dl
from dash import html

from thesis.common.config import BBOX


def create_user_layout() -> html.Div:
    """
    Create the user tab layout with interactive map and prediction controls.

    Returns:
        html.Div: Complete user tab layout.
    """
    bbox_bounds = [[BBOX[1], BBOX[0]], [BBOX[3], BBOX[2]]]
    return html.Div(
        [
            html.H3("Trip Prediction"),
            html.P("Move around the simulation map and click twice to select source and destination points."),
            html.Div(
                [
                    html.H4("Source"),
                    html.Div(
                        [
                            html.Div(["Latitude: ", html.Span("-", id="source-latitude")]),
                            html.Div(["Longitude: ", html.Span("-", id="source-longitude")]),
                        ]
                    ),
                    html.H4("Destination"),
                    html.Div(
                        [
                            html.Div(["Latitude: ", html.Span("-", id="destination-latitude")]),
                            html.Div(["Longitude: ", html.Span("-", id="destination-longitude")]),
                        ]
                    ),
                    html.Button("Clear Selection", id="button-clear-selection", n_clicks=0),
                    html.Button("Predict", id="button-predict", n_clicks=0, disabled=True),
                ]
            ),
            html.Div(
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
                            # TODO: fix markers not showing up and spans not filling up
                            dl.LayerGroup(id="user-markers"),
                        ],
                        style={
                            "maxWidth": "1000px",
                            "aspectRatio": "1.6",
                        },
                    ),
                ]
            ),
            html.Div(id="prediction-output"),
        ]
    )
