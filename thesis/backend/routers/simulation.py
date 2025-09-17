from fastapi import APIRouter, Request

from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.schemas import MetricsResponse, SimulationStatus

router = APIRouter()


# TODO: add response models
@router.post("/start", response_model=SimulationStatus)
async def start_simulation(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    return await mgr.start()


@router.post("/tick")
async def run_tick(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    return await mgr.manual_tick()


@router.post("/pause")
async def pause_simulation(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    return await mgr.pause()


@router.post("/resume")
async def resume_simulation(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    return await mgr.resume()


@router.post("/restart")
async def restart_simulation(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    await mgr.restart()
    store: MetricsStore = request.app.state.metrics_store
    store.clear()
    return {"status": "ok"}


@router.get("/status", response_model=SimulationStatus)
def get_status(request: Request):
    mgr: SimulationManager = request.app.state.simulation_manager
    return mgr.get_status()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    store: MetricsStore = request.app.state.metrics_store
    return store.get_all()
