import uvicorn
from fastapi import FastAPI

from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformService, PlatformServiceConfig
from thesis.drift.routers.drift import router as drift_router

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)

app = FastAPI(title="Platform Drift Service", version="1.0.0")
app.include_router(drift_router, prefix="/drift", tags=["drift"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    return HealthResponse(status="healthy", service=PlatformService.DRIFT)


if __name__ == "__main__":
    uvicorn.run("thesis.drift.main:app", host=config.host, port=config.port, reload=config.is_development)
