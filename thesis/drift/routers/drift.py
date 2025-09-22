from fastapi import APIRouter

from thesis.common.enums import DriftState
from thesis.common.schemas import DriftErrorsRequest, DriftErrorsResponse

drift_router = APIRouter()


@drift_router.post("/errors", response_model=DriftErrorsResponse)
def process_drift_errors(req: DriftErrorsRequest) -> DriftErrorsResponse:
    """
    Process drift errors.

    Args:
        req (DriftErrorsRequest): Request for drift errors.

    Returns:
        DriftErrorsResponse: Response for drift errors.
    """
    # TODO: implement actual drift lifecycle pipeline
    return DriftErrorsResponse(task=req.task, state=DriftState.STABLE, start_timestamp=0)
