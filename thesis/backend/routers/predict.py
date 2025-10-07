"""Prediction router for trip predictions."""

from fastapi import APIRouter, Request

from thesis.backend.services.prediction_service import PredictionService
from thesis.common.schemas import PredictionSingleRequest, TripPredictionResponse

predict_router = APIRouter()


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
