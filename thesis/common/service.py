"""Platform service configuration and utilities."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from thesis.common.config import (
    ACCESS_LOG,
    APPDATA_DIRNAME,
    COMMON_DIRNAME,
    DATA_DIRNAME,
    ENVIRONMENT,
    HOST,
    HTTP,
    LOGS_DIRNAME,
    LOOP,
    MISC_DIRNAME,
    MODELS_DIRNAME,
    PORT_BACKEND,
    PORT_DRIFT,
    PORT_FRONTEND,
    PORT_PREDICTOR_ETA,
    PORT_PREDICTOR_FUEL,
    PORT_PREDICTOR_STOPS,
    PROJECT_DIR,
)
from thesis.common.enums import MLTask, PlatformService


@dataclass(frozen=True, slots=True)
class PlatformServiceConfig:
    """A platform service config."""

    _PORTS: ClassVar[dict[PlatformService, int]] = {
        PlatformService.BACKEND: PORT_BACKEND,
        PlatformService.PREDICTOR_ETA: PORT_PREDICTOR_ETA,
        PlatformService.PREDICTOR_FUEL: PORT_PREDICTOR_FUEL,
        PlatformService.PREDICTOR_STOPS: PORT_PREDICTOR_STOPS,
        PlatformService.FRONTEND: PORT_FRONTEND,
        PlatformService.DRIFT: PORT_DRIFT,
    }

    _ML_TASKS: ClassVar[dict[PlatformService, MLTask | None]] = {
        PlatformService.BACKEND: None,
        PlatformService.PREDICTOR_ETA: MLTask.ETA,
        PlatformService.PREDICTOR_FUEL: MLTask.FUEL,
        PlatformService.PREDICTOR_STOPS: MLTask.STOPS,
        PlatformService.FRONTEND: None,
        PlatformService.DRIFT: None,
    }

    def __post_init__(self) -> None:
        for directory in [
            self.app_dir,
            self.common_dir,
            self.logs_dir,
            self.data_dir,
            self.models_dir,
            self.misc_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.service=}, "
            f"{self.port=}, "
            f"{self.ml_task=}, "
            f"{self.host=}, "
            f"{self.is_development=}, "
            f"{self.loop=}, "
            f"{self.http=}, "
            f"{self.access_log=}, "
            f"{self.app_dir=}, "
            f"{self.common_dir=}, "
            f"{self.logs_dir=}, "
            f"{self.data_dir=}, "
            f"{self.models_dir=}, "
            f"{self.misc_dir=}, "
            f"{self.backend_url=}, "
            f"{self.predictor_eta_url=}, "
            f"{self.predictor_fuel_url=}, "
            f"{self.predictor_stops_url=}, "
            f"{self.drift_url=})"
        )

    @property
    def service(self) -> PlatformService:
        """Platform service."""
        return PlatformService(os.environ.get("SERVICE", "backend"))

    @property
    def port(self) -> int:
        """Port for the platform service."""
        return PlatformServiceConfig._PORTS[self.service]

    @property
    def ml_task(self) -> MLTask | None:
        """ML task for the predictor if applicable."""
        return PlatformServiceConfig._ML_TASKS[self.service]

    @property
    def host(self) -> str:
        """Host for the platform service."""
        return HOST

    @property
    def is_development(self) -> bool:
        """Bool flag for development environment."""
        return self._environment == "development"

    @property
    def loop(self) -> str:
        """Event loop for the platform service."""
        return LOOP

    @property
    def http(self) -> str:
        """HTTP server for the platform service."""
        return HTTP

    @property
    def access_log(self) -> bool:
        """Bool access log flag for the platform service."""
        return ACCESS_LOG

    @property
    def app_dir(self) -> Path:
        """Path to the app directory."""
        return Path(os.environ.get("APP_DIR", PROJECT_DIR / APPDATA_DIRNAME))

    @property
    def common_dir(self) -> Path:
        """Path to the common directory."""
        return self.app_dir / COMMON_DIRNAME

    @property
    def logs_dir(self) -> Path:
        """Path to the logs directory."""
        return self.app_dir / LOGS_DIRNAME / self.service

    @property
    def data_dir(self) -> Path:
        """Path to the data directory, ML task-specific if applicable."""
        return self.app_dir / DATA_DIRNAME / self.ml_task if self.ml_task else self.app_dir / DATA_DIRNAME

    @property
    def models_dir(self) -> Path:
        """Path to the models directory, ML task-specific if applicable."""
        return self.app_dir / MODELS_DIRNAME / self.ml_task if self.ml_task else self.app_dir / MODELS_DIRNAME

    @property
    def misc_dir(self) -> Path:
        """Path to the misc directory, ML task-specific if applicable."""
        return self.app_dir / MISC_DIRNAME / self.ml_task if self.ml_task else self.app_dir / MISC_DIRNAME

    @property
    def backend_url(self) -> str:
        """URL to the backend service."""
        return f"http://{PlatformService.BACKEND}:{PORT_BACKEND}"

    @property
    def predictor_eta_url(self) -> str:
        """URL to the ETA predictor service."""
        return f"http://{PlatformService.PREDICTOR_ETA}:{PORT_PREDICTOR_ETA}"

    @property
    def predictor_fuel_url(self) -> str:
        """URL to the Fuel predictor service."""
        return f"http://{PlatformService.PREDICTOR_FUEL}:{PORT_PREDICTOR_FUEL}"

    @property
    def predictor_stops_url(self) -> str:
        """URL to the Stops predictor service."""
        return f"http://{PlatformService.PREDICTOR_STOPS}:{PORT_PREDICTOR_STOPS}"

    @property
    def drift_url(self) -> str:
        """URL to the Drift service."""
        return f"http://{PlatformService.DRIFT}:{PORT_DRIFT}"

    @property
    def _environment(self) -> str:
        """Environment for the platform service."""
        return os.environ.get("ENVIRONMENT", ENVIRONMENT)
