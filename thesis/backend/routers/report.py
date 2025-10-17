"""Report router for AI-generated simulation reports."""

from fastapi import APIRouter, Request

from thesis.backend.services.report_store import ReportStore
from thesis.common.schemas import ReportGenerationResponse, ReportStatusResponse

report_router = APIRouter()


@report_router.get("/status", response_model=ReportStatusResponse)
async def get_report_status(request: Request) -> ReportStatusResponse:
    """
    Get the status of report generation.

    Args:
        request (Request): Request object.

    Returns:
        ReportStatusResponse: Report status.
    """
    report_store: ReportStore = request.app.state.report_store
    return await report_store.get_status()


@report_router.get("/content", response_model=ReportGenerationResponse)
async def get_report_content(request: Request) -> ReportGenerationResponse:
    """
    Get the generated simulation report content.

    Args:
        request (Request): Request object.

    Returns:
        ReportGenerationResponse: Report content.
    """
    report_store: ReportStore = request.app.state.report_store
    return await report_store.get_content()
