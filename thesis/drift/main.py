from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from thesis.common.enums import PlatformService, PlatformServiceStatus
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformServiceConfig
from thesis.drift.routers.drift import drift_router
from thesis.drift.services.drift_service import DriftService

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        drift_service = DriftService()

        app.state.drift_service = drift_service
        yield
    finally:
        drift_service: DriftService = getattr(app.state, "drift_service", None)

        if drift_service is not None:
            await drift_service.clear()

        if hasattr(app.state, "drift_service"):
            delattr(app.state, "drift_service")


app = FastAPI(title="Platform Drift Service", version="1.0.0", lifespan=lifespan, default_response_class=ORJSONResponse)
app.include_router(drift_router, prefix="/drift", tags=["drift"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    return HealthResponse(status=PlatformServiceStatus.HEALTHY, service=PlatformService.DRIFT)


if __name__ == "__main__":
    uvicorn.run(
        "thesis.drift.main:app",
        host=config.host,
        port=config.port,
        reload=config.is_development,
        loop=config.loop,
        http=config.http,
        access_log=config.access_log,
    )
