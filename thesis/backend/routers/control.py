"""Control router for simulation lifecycle management."""

from fastapi import APIRouter, Request

from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.schemas import SimulationSnapshot

control_router = APIRouter()


@control_router.post("/start", response_model=SimulationSnapshot)
async def start_simulation(request: Request) -> SimulationSnapshot:
    """
    Start the simulation.

    Args:
        request (Request): Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.start()


@control_router.post("/pause", response_model=SimulationSnapshot)
async def pause_simulation(request: Request) -> SimulationSnapshot:
    """
    Pause the simulation.

    Args:
        request (Request): Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.pause()


@control_router.post("/resume", response_model=SimulationSnapshot)
async def resume_simulation(request: Request) -> SimulationSnapshot:
    """
    Resume the simulation.

    Args:
        request (Request): Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.resume()


@control_router.post("/reset", response_model=SimulationSnapshot)
async def reset_simulation(request: Request) -> SimulationSnapshot:
    """
    Reset the simulation.

    Args:
        request (Request): Request object.

    Returns:
        SimulationSnapshot: Current snapshot of the simulation.
    """
    simulation_manager: SimulationManager = request.app.state.simulation_manager
    return await simulation_manager.reset()
