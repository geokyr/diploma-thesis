from fastapi import APIRouter, Request

from thesis.common.enums import MLTask
from thesis.common.schemas import DriftErrorsRequest, DriftErrorsResponse

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
    # TODO: implement actual drift lifecycle pipeline
    state_service = request.app.state.state_service
    return state_service.get_state(req.ml_task)


@drift_router.get("/status")
async def get_drift_status(ml_task: MLTask, request: Request) -> DriftErrorsResponse:
    """
    Get current drift status for an ML task.

    Args:
        ml_task (MLTask): ML task.

    Returns:
        DriftErrorsResponse: Drift status for the ML task.
    """
    state_service = request.app.state.state_service
    return state_service.get_state(ml_task)
