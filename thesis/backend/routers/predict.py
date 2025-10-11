"""Prediction router for trip predictions."""

from fastapi import APIRouter, Request

from thesis.backend.services.prediction_service import PredictionService
from thesis.common.schemas import (
    PredictionSingleRequest,
    RoutePreviewRequest,
    RoutePreviewResponse,
    TripPredictionResponse,
)

predict_router = APIRouter()


@predict_router.post("/preview", response_model=RoutePreviewResponse)
async def preview_route(preview_request: RoutePreviewRequest, request: Request) -> RoutePreviewResponse:
    """
    Get route preview polyline for the given source and destination.

    Args:
        preview_request (RoutePreviewRequest): Route preview request.
        request (Request): Request object.

    Returns:
        RoutePreviewResponse: Route polyline as list of (lat, lon) tuples
    """
    prediction_service: PredictionService = request.app.state.prediction_service
    return await prediction_service.preview_route(preview_request)


@predict_router.post("/trip", response_model=TripPredictionResponse)
async def predict_trip(trip_request: PredictionSingleRequest, request: Request) -> TripPredictionResponse:
    """
    Predict trip metrics for all available ML tasks.

    Args:
        trip_request (PredictionSingleRequest): Trip prediction request.
        request (Request): Request object.

    Returns:
        TripPredictionResponse: Dictionary of predictions per available ML task
    """
    prediction_service: PredictionService = request.app.state.prediction_service
    return await prediction_service.predict_trip(trip_request)
