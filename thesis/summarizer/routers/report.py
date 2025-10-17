"""Report router."""

from fastapi import APIRouter, Request

from thesis.common.schemas import ReportGenerationRequest, ReportGenerationResponse
from thesis.summarizer.services.report_generator import ReportGenerator

report_router = APIRouter()


@report_router.post("/generate", response_model=ReportGenerationResponse)
async def generate_report(req: ReportGenerationRequest, request: Request) -> ReportGenerationResponse:
    """
    Generate a markdown report from simulation data.

    Args:
        req (ReportGenerationRequest): Request containing notifications and metrics.
        request (Request): FastAPI request.

    Returns:
        ReportGenerationResponse: Generated markdown report.
    """
    report_generator: ReportGenerator = request.app.state.report_generator
    return await report_generator.generate_report(req.notifications, req.metrics)
