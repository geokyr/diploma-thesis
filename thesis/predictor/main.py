from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformService, PlatformServiceConfig
from thesis.predictor.routers.predict import router as predict_router
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.predictor import Predictor

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        manager = ModelManager(config)
        manager.load_default()
        predictor = Predictor(manager.model, config)
        app.state.model_manager = manager
        app.state.predictor = predictor
        yield
    finally:
        app.state.model_manager = None
        app.state.predictor = None


app = FastAPI(title="Platform Predictor Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(status="healthy", service=PlatformService.PREDICTOR_ETA)


app.include_router(predict_router, prefix="/predict", tags=["predict"])


if __name__ == "__main__":
    uvicorn.run("thesis.predictor.main:app", host=config.host, port=config.port, reload=config.is_development)
