from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from thesis.backend.routers.control import control_router
from thesis.backend.routers.predict import predict_router
from thesis.backend.routers.simulation import simulation_router
from thesis.backend.services.metrics_store import MetricsStore
from thesis.backend.services.notification_store import NotificationStore
from thesis.backend.services.prediction_service import PredictionService
from thesis.backend.services.simulation_manager import SimulationManager
from thesis.backend.services.timelapse_driver import TimelapseDriver
from thesis.common.enums import PlatformService, PlatformServiceStatus
from thesis.common.logger import setup_logger
from thesis.common.schemas import HealthResponse
from thesis.common.service import PlatformServiceConfig

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        metrics_store = MetricsStore()
        notification_store = NotificationStore()
        timelapse_driver = TimelapseDriver(
            config=config, metrics_store=metrics_store, notification_store=notification_store
        )
        simulation_manager = SimulationManager(timelapse_driver=timelapse_driver)
        prediction_service = PredictionService(timelapse_driver=timelapse_driver)

        app.state.metrics_store = metrics_store
        app.state.notification_store = notification_store
        app.state.timelapse_driver = timelapse_driver
        app.state.simulation_manager = simulation_manager
        app.state.prediction_service = prediction_service

        await timelapse_driver.detect_available_ml_tasks()
        yield
    finally:
        prediction_service: PredictionService = getattr(app.state, "prediction_service", None)
        simulation_manager: SimulationManager = getattr(app.state, "simulation_manager", None)
        timelapse_driver: TimelapseDriver = getattr(app.state, "timelapse_driver", None)
        notification_store: NotificationStore = getattr(app.state, "notification_store", None)
        metrics_store: MetricsStore = getattr(app.state, "metrics_store", None)

        if prediction_service is not None:
            await prediction_service.clear()
        if simulation_manager is not None:
            await simulation_manager.clear()
        if timelapse_driver is not None:
            await timelapse_driver.clear()
        if notification_store is not None:
            await notification_store.clear()
        if metrics_store is not None:
            await metrics_store.clear()

        if hasattr(app.state, "prediction_service"):
            delattr(app.state, "prediction_service")
        if hasattr(app.state, "simulation_manager"):
            delattr(app.state, "simulation_manager")
        if hasattr(app.state, "timelapse_driver"):
            delattr(app.state, "timelapse_driver")
        if hasattr(app.state, "notification_store"):
            delattr(app.state, "notification_store")
        if hasattr(app.state, "metrics_store"):
            delattr(app.state, "metrics_store")


app = FastAPI(title="Platform Backend API", version="1.0.0", lifespan=lifespan)
app.include_router(control_router, prefix="/control", tags=["control"])
app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
app.include_router(predict_router, prefix="/predict", tags=["predict"])


@app.get("/health", response_model=HealthResponse, tags=["health"])
def get_health() -> HealthResponse:
    return HealthResponse(status=PlatformServiceStatus.HEALTHY, service=PlatformService.BACKEND)


if __name__ == "__main__":
    uvicorn.run("thesis.backend.main:app", host=config.host, port=config.port, reload=config.is_development)
