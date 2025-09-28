"""Simulation router."""

from fastapi import APIRouter, Request

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.enums import MLTask
from thesis.common.schemas import MetricsResponse, SimulationSnapshot

simulation_router = APIRouter()


@simulation_router.post("/start", response_model=SimulationSnapshot)
async def start_simulation(request: Request) -> SimulationSnapshot:
    """
    Start the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.start()


@simulation_router.post("/pause", response_model=SimulationSnapshot)
async def pause_simulation(request: Request) -> SimulationSnapshot:
    """
    Pause the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.pause()


@simulation_router.post("/resume", response_model=SimulationSnapshot)
async def resume_simulation(request: Request) -> SimulationSnapshot:
    """
    Resume the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.resume()


@simulation_router.post("/reset", response_model=SimulationSnapshot)
async def reset_simulation(request: Request) -> SimulationSnapshot:
    """
    Reset the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.reset()


@simulation_router.get("/snapshot", response_model=SimulationSnapshot)
async def get_simulation_snapshot(request: Request) -> SimulationSnapshot:
    """
    Get the current snapshot of the simulation.

    Args:
        request: Request object.

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
        request: Request object.

    Returns:
        MetricsResponse: Metrics of the simulation.
    """
    metrics_store: MetricsStore = request.app.state.metrics_store
    return await metrics_store.get_metrics(ml_task)
