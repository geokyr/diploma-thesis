"""Retraining endpoints and job orchestration for predictor."""

from fastapi import APIRouter, Request

from thesis.common.schemas import RetrainRequest, RetrainResponse, RetrainStatusResponse
from thesis.predictor.services.retrain_service import RetrainService

retrain_router = APIRouter()


@retrain_router.post("/start", response_model=RetrainResponse)
def start_retrain(req: RetrainRequest, request: Request) -> RetrainResponse:
    """
    Start retraining for a given window.

    Args:
        req (RetrainRequest): Request for retraining.
        request (Request): FastAPI request.

    Returns:
        RetrainResult: Result for retraining.
    """
    retrain_service: RetrainService = request.app.state.retrain_service
    job_id = retrain_service.start(req.start_timestamp, req.end_timestamp)
    return RetrainResponse(job_id=job_id)


@retrain_router.get("/status/{job_id}", response_model=RetrainStatusResponse)
def retrain_status(job_id: str, request: Request) -> RetrainStatusResponse:
    """
    Get status of a retraining job.

    Args:
        job_id (str): Job ID.
        request (Request): FastAPI request.

    Returns:
        RetrainStatusResponse: Status of the job.
    """
    retrain_service: RetrainService = request.app.state.retrain_service
    status = retrain_service.get_status(job_id)
    return RetrainStatusResponse(status=status)
