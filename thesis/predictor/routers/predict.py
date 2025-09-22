"""Predict router."""

from fastapi import APIRouter, Request

from thesis.common.schemas import PredictionBatchRequest, PredictionBatchResponse
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

    return predictor.predict_window(
        req.start_timestamp,
        req.end_timestamp,
    )
