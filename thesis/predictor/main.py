from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from thesis.common.enums import PlatformServiceStatus
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformService, PlatformServiceConfig
from thesis.predictor.routers.predict import predict_router
from thesis.predictor.routers.retrain import retrain_router
from thesis.predictor.services.data_loader import DataLoader
from thesis.predictor.services.model_manager import ModelManager
from thesis.predictor.services.predictor import Predictor
from thesis.predictor.services.retrain_service import RetrainService
from thesis.predictor.services.sumo_service import SumoService

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        sumo_service = SumoService()
        data_loader = DataLoader(data_dir=config.data_dir)
        model_manager = ModelManager(models_dir=config.models_dir)
        retrain_service = RetrainService(data_dir=config.data_dir, model_manager=model_manager)
        predictor = Predictor(
            ml_task=config.ml_task,
            misc_dir=config.misc_dir,
            data_loader=data_loader,
            model_manager=model_manager,
            sumo_service=sumo_service,
        )

        app.state.sumo_service = sumo_service
        app.state.data_loader = data_loader
        app.state.model_manager = model_manager
        app.state.retrain_service = retrain_service
        app.state.predictor = predictor
        yield
    finally:
        predictor: Predictor = getattr(app.state, "predictor", None)
        retrain_service: RetrainService = getattr(app.state, "retrain_service", None)
        model_manager: ModelManager = getattr(app.state, "model_manager", None)
        data_loader: DataLoader = getattr(app.state, "data_loader", None)
        sumo_service: SumoService = getattr(app.state, "sumo_service", None)

        if predictor is not None:
            predictor.clear()
        if retrain_service is not None:
            retrain_service.clear()
        if model_manager is not None:
            model_manager.clear()
        if data_loader is not None:
            data_loader.clear()
        if sumo_service is not None:
            sumo_service.clear()

        if hasattr(app.state, "predictor"):
            delattr(app.state, "predictor")
        if hasattr(app.state, "retrain_service"):
            delattr(app.state, "retrain_service")
        if hasattr(app.state, "model_manager"):
            delattr(app.state, "model_manager")
        if hasattr(app.state, "data_loader"):
            delattr(app.state, "data_loader")
        if hasattr(app.state, "sumo_service"):
            delattr(app.state, "sumo_service")


app = FastAPI(title="Platform Predictor Service", version="1.0.0", lifespan=lifespan)
app.include_router(predict_router, prefix="/predict", tags=["predict"])
app.include_router(retrain_router, prefix="/retrain", tags=["retrain"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def get_health() -> HealthResponse:
    return HealthResponse(status=PlatformServiceStatus.HEALTHY, service=PlatformService.PREDICTOR_ETA)


if __name__ == "__main__":
    uvicorn.run("thesis.predictor.main:app", host=config.host, port=config.port, reload=config.is_development)
