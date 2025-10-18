"""Simulation router for monitoring and data access."""

from fastapi import APIRouter, Request

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.notification_store import NotificationStore
from thesis.backend.services.report_store import ReportStore
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.enums import MLTask
from thesis.common.schemas import MetricsResponse, NotificationFeed, ReportResponse, SimulationSnapshot

simulation_router = APIRouter()


@simulation_router.get("/snapshot", response_model=SimulationSnapshot)
async def get_simulation_snapshot(request: Request) -> SimulationSnapshot:
    """
    Get the current snapshot of the simulation.

    Args:
        request (Request): Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.get_snapshot()


@simulation_router.get("/metrics", response_model=MetricsResponse)
async def get_simulation_metrics(ml_task: MLTask, request: Request) -> MetricsResponse:
    """
    Get the metrics of the simulation.

    Args:
        ml_task (MLTask): ML task.
        request (Request): Request object.

    Returns:
        MetricsResponse: Metrics of the simulation.
    """
    metrics_store: MetricsStore = request.app.state.metrics_store
    return await metrics_store.get_metrics(ml_task)


@simulation_router.get("/notifications", response_model=NotificationFeed)
async def get_simulation_notifications(request: Request) -> NotificationFeed:
    """
    Get all notifications of the simulation.

    Args:
        request (Request): Request object.

    Returns:
        NotificationFeed: Feed of all notifications.
    """
    notification_store: NotificationStore = request.app.state.notification_store
    return await notification_store.get_all()


@simulation_router.get("/report", response_model=ReportResponse)
async def get_simulation_report(request: Request) -> ReportResponse:
    """
    Get the simulation report status and content, if available.

    Args:
        request (Request): Request object.

    Returns:
        ReportResponse: Simulation report status and content.
    """
    report_store: ReportStore = request.app.state.report_store
    return await report_store.get_report()
