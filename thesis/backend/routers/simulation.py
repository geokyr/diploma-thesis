from fastapi import APIRouter, Request

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.schemas import MetricsResponse, SimulationSnapshot

simulation_router = APIRouter()


@simulation_router.post("/start", response_model=SimulationSnapshot)
async def start_simulation(request: Request):
    """
    Start the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: The current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.start_simulation()


@simulation_router.post("/pause", response_model=SimulationSnapshot)
async def pause_simulation(request: Request):
    """
    Pause the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: The current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.pause_simulation()


@simulation_router.post("/resume", response_model=SimulationSnapshot)
async def resume_simulation(request: Request):
    """
    Resume the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: The current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.resume_simulation()


@simulation_router.post("/restart", response_model=SimulationSnapshot)
async def restart_simulation(request: Request):
    """
    Restart the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: The current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    metrics_store: MetricsStore = request.app.state.metrics_store
    metrics_store.clear()
    return await simulation_manager.restart_simulation()


@simulation_router.get("/snapshot", response_model=SimulationSnapshot)
def get_snapshot(request: Request):
    """
    Get the current snapshot of the simulation.

    Args:
        request: Request object.

    Returns:
        SimulationSnapshot: The current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return simulation_manager.get_snapshot()


@simulation_router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    """
    Get the metrics of the simulation.

    Args:
        request: Request object.

    Returns:
        MetricsResponse: The metrics of the simulation.
    """
    metrics_store: MetricsStore = request.app.state.metrics_store
    return metrics_store.get_metrics()
