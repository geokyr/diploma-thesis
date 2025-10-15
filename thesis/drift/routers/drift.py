"""Drift router."""

from fastapi import APIRouter, Request

from thesis.common.enums import MLTask
from thesis.common.schemas import (
    DriftErrorsRequest,
    DriftErrorsResponse,
    DriftResetRequest,
    DriftResetResponse,
    RecalibrateRequest,
    RecalibrateResponse,
)
from thesis.drift.services.drift_service import DriftService

drift_router = APIRouter()


@drift_router.post("/errors", response_model=DriftErrorsResponse)
async def process_drift_errors(req: DriftErrorsRequest, request: Request) -> DriftErrorsResponse:
    """
    Process drift errors.

    Args:
        req (DriftErrorsRequest): Request for drift errors.

    Returns:
        DriftErrorsResponse: Response for drift errors.
    """
    drift_service: DriftService = request.app.state.drift_service
    return await drift_service.process_errors(req.ml_task, req.error_points)


@drift_router.get("/status")
async def get_drift_status(ml_task: MLTask, request: Request) -> DriftErrorsResponse:
    """
    Get current drift status for an ML task.

    Args:
        ml_task (MLTask): ML task.

    Returns:
        DriftErrorsResponse: Drift status for the ML task.
    """
    drift_service: DriftService = request.app.state.drift_service
    return await drift_service.get_state(ml_task)


@drift_router.post("/recalibrate", response_model=RecalibrateResponse)
async def recalibrate_detectors(req: RecalibrateRequest, request: Request) -> RecalibrateResponse:
    """
    Recalibrate drift detectors after model adaptation.

    Args:
        req (RecalibrateRequest): Request containing ML task.
        request (Request): FastAPI request.

    Returns:
        RecalibrateResponse: Recalibration success status.
    """
    drift_service: DriftService = request.app.state.drift_service
    return await drift_service.recalibrate_task(req.ml_task)


@drift_router.post("/reset", response_model=DriftResetResponse)
async def reset_drift_service(req: DriftResetRequest, request: Request) -> DriftResetResponse:
    """
    Reset drift detection for specified ML tasks.

    Args:
        req (DriftResetRequest): Request containing ML tasks to reset.
        request (Request): FastAPI request.

    Returns:
        DriftResetResponse: Response containing success status.
    """
    drift_service: DriftService = request.app.state.drift_service
    return await drift_service.reset_tasks(req.ml_tasks)
