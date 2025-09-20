from fastapi import APIRouter, Request

from thesis.common.schemas import PredictionBatchRequest, PredictionBatchResponse
from thesis.predictor.services.predictor import Predictor

router = APIRouter()


@router.post("/batch", response_model=PredictionBatchResponse)
def predict_batch(req: PredictionBatchRequest, request: Request) -> PredictionBatchResponse:
    predictor: Predictor = request.app.state.predictor

    return predictor.predict_window(
        req.time_window.start_timestamp,
        req.time_window.end_timestamp,
    )
