"""Predict router."""

from fastapi import APIRouter, Request

from thesis.common.schemas import (
    PredictionBatchRequest,
    PredictionBatchResponse,
    PredictionSingleRequest,
    PredictionSingleResponse,
    RoutePreviewRequest,
    RoutePreviewResponse,
)
from thesis.predictor.services.predictor import Predictor

predict_router = APIRouter()


@predict_router.post("/batch", response_model=PredictionBatchResponse)
def predict_batch(req: PredictionBatchRequest, request: Request) -> PredictionBatchResponse:
    """
    Predict a batch of data.

    Args:
        req (PredictionBatchRequest): Request for batch predictions.
        request (Request): FastAPI request.

    Returns:
        PredictionBatchResponse: Response for batch predictions.
    """
    predictor: Predictor = request.app.state.predictor
    return predictor.predict_window(req.start_timestamp, req.end_timestamp)


@predict_router.post("/single", response_model=PredictionSingleResponse)
def predict_single(req: PredictionSingleRequest, request: Request) -> PredictionSingleResponse:
    """
    Predict a single trip.

    Args:
        req (PredictionSingleRequest): Request for single trip prediction.
        request (Request): FastAPI request.

    Returns:
        PredictionSingleResponse: Response for single trip prediction.
    """
    predictor: Predictor = request.app.state.predictor
    return predictor.predict_single(
        req.start_timestamp,
        req.source_x,
        req.source_y,
        req.destination_x,
        req.destination_y,
        req.distance,
        req.edges,
        req.minimum_x,
        req.maximum_x,
        req.minimum_y,
        req.maximum_y,
    )


@predict_router.post("/route", response_model=RoutePreviewResponse)
def get_route(req: RoutePreviewRequest, request: Request) -> RoutePreviewResponse:
    """
    Get route polyline for the given source and destination.

    Args:
        req (RoutePreviewRequest): Request for route preview.
        request (Request): FastAPI request.

    Returns:
        RoutePreviewResponse: Route polyline as list of (lat, lon) tuples.
    """
    predictor: Predictor = request.app.state.predictor
    return predictor.get_route(
        req.source_latitude,
        req.source_longitude,
        req.destination_latitude,
        req.destination_longitude,
    )
