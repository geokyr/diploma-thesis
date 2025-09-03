"""
Platform service configuration and utilities.
Provides platform service types (backend/frontend/predictor-eta/predictor-fuel/predictor-stops) and configuration classes for managing service-specific settings and directories.
"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from thesis.common.config import (
    DATA_DIRNAME,
    DEBUG,
    HOST,
    LOGS_DIRNAME,
    MODELS_DIRNAME,
    PLATFORM_DIR,
    PORT_BACKEND,
    PORT_FRONTEND,
    PORT_PREDICTOR_ETA,
    PORT_PREDICTOR_FUEL,
    PORT_PREDICTOR_STOPS,
    STATE_DIRNAME,
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
        FRONTEND: Frontend service.
        PREDICTOR_ETA: ETA predictor service.
        PREDICTOR_FUEL: Fuel predictor service.
        PREDICTOR_STOPS: Stops predictor service.
    """

    BACKEND = "backend"
    FRONTEND = "frontend"
    PREDICTOR_ETA = "predictor-eta"
    PREDICTOR_FUEL = "predictor-fuel"
    PREDICTOR_STOPS = "predictor-stops"


@dataclass(frozen=True, slots=True)
class PlatformServiceConfig:
    """
    A platform service config.

    Properties:
        service (PlatformService): Service.
        port (int): Port.
        logs_dir (Path): Path to the logs directory.
        task (Task | None): Task.
        host (str): Host.
        debug (bool): Debug.
        backend_url (str): Backend URL.
        frontend_url (str): Frontend URL.
        predictor_eta_url (str): ETA predictor URL.
        predictor_fuel_url (str): Fuel predictor URL.
        predictor_stops_url (str): Stops predictor URL.
        data_dir (Path): Path to the data directory.
        models_dir (Path): Path to the models directory.
        state_dir (Path): Path to the state directory.
    """

    _PORTS = {
        PlatformService.BACKEND: PORT_BACKEND,
        PlatformService.FRONTEND: PORT_FRONTEND,
        PlatformService.PREDICTOR_ETA: PORT_PREDICTOR_ETA,
        PlatformService.PREDICTOR_FUEL: PORT_PREDICTOR_FUEL,
        PlatformService.PREDICTOR_STOPS: PORT_PREDICTOR_STOPS,
    }

    _TASKS = {
        PlatformService.BACKEND: None,
        PlatformService.FRONTEND: None,
        PlatformService.PREDICTOR_ETA: Task.ETA,
        PlatformService.PREDICTOR_FUEL: Task.FUEL,
        PlatformService.PREDICTOR_STOPS: Task.STOPS,
    }

    def __post_init__(self) -> None:
        for dir in [self.logs_dir, self.data_dir, self.models_dir, self.state_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.service=}, "
            f"{self.port=}, "
            f"{self.logs_dir=}, "
            f"{self.task=}, "
            f"{self.host=}, "
            f"{self.debug=}, "
            f"{self.backend_url=}, "
            f"{self.frontend_url=}, "
            f"{self.predictor_eta_url=}, "
            f"{self.predictor_fuel_url=}, "
            f"{self.predictor_stops_url=}, "
            f"{self.data_dir=}, "
            f"{self.models_dir=}, "
            f"{self.state_dir=})"
        )

    @property
    def service(self) -> PlatformService:
        return PlatformService(os.environ.get("SERVICE"))

    @property
    def port(self) -> int:
        return self._PORTS[self.service]

    @property
    def logs_dir(self) -> Path:
        return PLATFORM_DIR / LOGS_DIRNAME / self.service

    @property
    def task(self) -> Task | None:
        return self._TASKS[self.service]

    @property
    def host(self) -> str:
        return HOST

    @property
    def debug(self) -> bool:
        return DEBUG

    @property
    def backend_url(self) -> str:
        return f"http://{PlatformService.BACKEND}:{PORT_BACKEND}"

    @property
    def frontend_url(self) -> str:
        return f"http://{PlatformService.FRONTEND}:{PORT_FRONTEND}"

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
    def data_dir(self) -> Path:
        return PLATFORM_DIR / DATA_DIRNAME

    @property
    def models_dir(self) -> Path:
        return PLATFORM_DIR / MODELS_DIRNAME

    @property
    def state_dir(self) -> Path:
        return PLATFORM_DIR / STATE_DIRNAME
