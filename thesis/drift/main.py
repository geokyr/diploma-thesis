from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from thesis.common.enums import PlatformServiceStatus
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformService, PlatformServiceConfig
from thesis.drift.routers.drift import drift_router
from thesis.drift.services.state_service import StateService

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state_service = StateService()

        app.state.state_service = state_service
        yield
    finally:
        state_service: StateService = getattr(app.state, "state_service", None)

        if state_service is not None:
            state_service.clear()

        if hasattr(app.state, "state_service"):
            delattr(app.state, "state_service")


app = FastAPI(title="Platform Drift Service", version="1.0.0", lifespan=lifespan)
app.include_router(drift_router, prefix="/drift", tags=["drift"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def get_health() -> HealthResponse:
    return HealthResponse(status=PlatformServiceStatus.HEALTHY, service=PlatformService.DRIFT)


if __name__ == "__main__":
    uvicorn.run("thesis.drift.main:app", host=config.host, port=config.port, reload=config.is_development)
