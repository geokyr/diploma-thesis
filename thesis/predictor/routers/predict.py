"""Predict router."""

from fastapi import APIRouter, Request

from thesis.common.schemas import PredictionBatchRequest, PredictionBatchResponse
from thesis.predictor.services.predictor import Predictor

predict_router = APIRouter()


@predict_router.post("/batch", response_model=PredictionBatchResponse)
def predict_batch(req: PredictionBatchRequest, request: Request) -> PredictionBatchResponse:
    predictor: Predictor = request.app.state.predictor

    return predictor.predict_window(
        req.start_timestamp,
        req.end_timestamp,
    )
