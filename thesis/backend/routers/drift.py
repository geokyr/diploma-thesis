"""Drift router."""

from fastapi import APIRouter, Request

from thesis.backend.services.drift_client import DriftClient
from thesis.common.schemas import DriftErrorsResponse

drift_router = APIRouter()


@drift_router.get("/status", response_model=DriftErrorsResponse)
async def get_drift_status(request: Request) -> DriftErrorsResponse:
    """
    Get the status of all ML tasks from drift service.

    Args:
        request: Request object.

    Returns:
        DriftErrorsResponse: Status of all ML tasks.
    """
    drift_client: DriftClient = request.app.state.drift_client
    return await drift_client.get_status()
