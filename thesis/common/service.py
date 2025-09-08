"""
Platform service configuration and utilities.
Provides platform service types (backend/predictor-eta/predictor-fuel/predictor-stops/frontend/drift/redis) and configuration classes for managing service-specific settings and directories.
"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from thesis.common.config import (
    APPDATA_DIRNAME,
    DATA_DIRNAME,
    ENVIRONMENT,
    HOST,
    LOGS_DIRNAME,
    MODELS_DIRNAME,
    PORT_BACKEND,
    PORT_DRIFT,
    PORT_FRONTEND,
    PORT_PREDICTOR_ETA,
    PORT_PREDICTOR_FUEL,
    PORT_PREDICTOR_STOPS,
    PORT_REDIS,
    PROJECT_DIR,
)


class Task(StrEnum):
    """
    Tasks.

    Attributes:
        ETA: Estimated time of arrival prediction task.
        FUEL: Fuel consumption prediction task.
        STOPS: Number of stops prediction task.
    """

    ETA = "eta"
    FUEL = "fuel"
    STOPS = "stops"


class PlatformService(StrEnum):
    """
    Platform services.

    Attributes:
        BACKEND: Backend service.
        PREDICTOR_ETA: ETA predictor service.
        PREDICTOR_FUEL: Fuel predictor service.
        PREDICTOR_STOPS: Stops predictor service.
        FRONTEND: Frontend service.
        DRIFT: Drift service.
        REDIS: Redis service.
    """

    BACKEND = "backend"
    PREDICTOR_ETA = "predictor-eta"
    PREDICTOR_FUEL = "predictor-fuel"
    PREDICTOR_STOPS = "predictor-stops"
    FRONTEND = "frontend"
    DRIFT = "drift"
    REDIS = "redis"


@dataclass(frozen=True, slots=True)
class PlatformServiceConfig:
    """
    A platform service config.

    Properties:
        service (PlatformService): Service.
        port (int): Port.
        task (Task | None): Task.
        host (str): Host.
        is_development (bool): Is development.
        app_dir (Path): Path to the app directory.
        logs_dir (Path): Path to the logs directory.
        data_dir (Path): Path to the data directory.
        models_dir (Path): Path to the models directory.
        backend_url (str): Backend URL.
        predictor_eta_url (str): ETA predictor URL.
        predictor_fuel_url (str): Fuel predictor URL.
        predictor_stops_url (str): Stops predictor URL.
        frontend_url (str): Frontend URL.
        drift_url (str): Drift URL.
        redis_url (str): Redis URL.
    """

    _PORTS = {
        PlatformService.BACKEND: PORT_BACKEND,
        PlatformService.PREDICTOR_ETA: PORT_PREDICTOR_ETA,
        PlatformService.PREDICTOR_FUEL: PORT_PREDICTOR_FUEL,
        PlatformService.PREDICTOR_STOPS: PORT_PREDICTOR_STOPS,
        PlatformService.FRONTEND: PORT_FRONTEND,
        PlatformService.DRIFT: PORT_DRIFT,
        PlatformService.REDIS: PORT_REDIS,
    }

    _TASKS = {
        PlatformService.BACKEND: None,
        PlatformService.PREDICTOR_ETA: Task.ETA,
        PlatformService.PREDICTOR_FUEL: Task.FUEL,
        PlatformService.PREDICTOR_STOPS: Task.STOPS,
        PlatformService.FRONTEND: None,
        PlatformService.DRIFT: None,
        PlatformService.REDIS: None,
    }

    def __post_init__(self) -> None:
        for dir in [self.app_dir, self.logs_dir, self.data_dir, self.models_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.service=}, "
            f"{self.port=}, "
            f"{self.task=}, "
            f"{self.host=}, "
            f"{self.is_development=}, "
            f"{self.app_dir=}, "
            f"{self.logs_dir=}, "
            f"{self.data_dir=}, "
            f"{self.models_dir=}, "
            f"{self.backend_url=}, "
            f"{self.predictor_eta_url=}, "
            f"{self.predictor_fuel_url=}, "
            f"{self.predictor_stops_url=}, "
            f"{self.frontend_url=}, "
            f"{self.drift_url=}, "
            f"{self.redis_url=})"
        )

    @property
    def service(self) -> PlatformService:
        return PlatformService(os.environ.get("SERVICE", "backend"))

    @property
    def port(self) -> int:
        return self._PORTS[self.service]

    @property
    def task(self) -> Task | None:
        return self._TASKS[self.service]

    @property
    def host(self) -> str:
        return HOST

    @property
    def is_development(self) -> bool:
        return self._environment == "development"

    @property
    def app_dir(self) -> Path:
        return Path(os.environ.get("APP_DIR", PROJECT_DIR / APPDATA_DIRNAME))

    @property
    def logs_dir(self) -> Path:
        return self.app_dir / LOGS_DIRNAME / self.service

    @property
    def data_dir(self) -> Path:
        return self.app_dir / DATA_DIRNAME

    @property
    def models_dir(self) -> Path:
        return self.app_dir / MODELS_DIRNAME

    @property
    def backend_url(self) -> str:
        return f"http://{PlatformService.BACKEND}:{PORT_BACKEND}"

    @property
    def predictor_eta_url(self) -> str:
        return f"http://{PlatformService.PREDICTOR_ETA}:{PORT_PREDICTOR_ETA}"

    @property
    def predictor_fuel_url(self) -> str:
        return f"http://{PlatformService.PREDICTOR_FUEL}:{PORT_PREDICTOR_FUEL}"

    @property
    def predictor_stops_url(self) -> str:
        return f"http://{PlatformService.PREDICTOR_STOPS}:{PORT_PREDICTOR_STOPS}"

    @property
    def frontend_url(self) -> str:
        return f"http://{PlatformService.FRONTEND}:{PORT_FRONTEND}"

    @property
    def drift_url(self) -> str:
        return f"http://{PlatformService.DRIFT}:{PORT_DRIFT}"

    @property
    def redis_url(self) -> str:
        return f"http://{PlatformService.REDIS}:{PORT_REDIS}"

    @property
    def _environment(self) -> str:
        return os.environ.get("ENVIRONMENT", ENVIRONMENT)
