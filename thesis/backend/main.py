from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from thesis.backend.routers.metrics import router as metrics_router
from thesis.backend.routers.simulation import router as simulation_router
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformService, PlatformServiceConfig

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.simulation_manager = SimulationManager()
        yield
    finally:
        simulation_manager = getattr(app.state, "simulation_manager", None)
        if simulation_manager is not None:
            await simulation_manager.shutdown()
        if hasattr(app.state, "simulation_manager"):
            delattr(app.state, "simulation_manager")


app = FastAPI(title="Platform Backend API", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy", service=PlatformService.BACKEND)


app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])

if __name__ == "__main__":
    uvicorn.run("thesis.backend.main:app", host=config.host, port=config.port, reload=config.is_development)
