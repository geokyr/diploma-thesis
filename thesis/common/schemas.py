"""Shared Pydantic schemas for API models across all platform services."""

from datetime import datetime

from pydantic import BaseModel, Field

from thesis.common.enums import DriftState, MLTask, SimulationState
from thesis.common.service import PlatformService


class TimeWindow(BaseModel):
    """Time window for predictions or retraining."""

    start_timestamp: float = Field(..., description="Start timestamp of the window")
    end_timestamp: float = Field(..., description="End timestamp of the window")


class ErrorPoint(BaseModel):
    """Error point for predictions or retraining."""

    timestamp: float = Field(..., description="Timestamp of the error")
    error: float = Field(..., description="Error value")


class PredictionRequest(BaseModel):
    """Request for predictions."""

    time_window: TimeWindow = Field(..., description="Time window for predictions")


class PredictionResponse(BaseModel):
    """Response for predictions."""

    points: list[ErrorPoint] = Field(..., description="List of error points")


class DriftErrorsRequest(BaseModel):
    """Request for drift errors."""

    task: MLTask = Field(..., description="ML task")
    points: list[ErrorPoint] = Field(..., description="List of error points")


class DriftErrorsResponse(BaseModel):
    """Response for drift errors."""

    task: MLTask = Field(..., description="ML task")
    state: DriftState = Field(..., description="Drift state")
    start_timestamp: float = Field(..., description="Start timestamp of the state")


class RetrainRequest(BaseModel):
    """Request for retraining."""

    time_window: TimeWindow = Field(..., description="Time window for retraining")


class RetrainResult(BaseModel):
    """Result for retraining."""

    job_id: str = Field(..., description="Job ID")
    version: str = Field(..., description="Version of the model")


class RetrainStatusRequest(BaseModel):
    """Request for retraining status."""

    job_id: str = Field(..., description="Job ID")


class RetrainStatusResponse(BaseModel):
    """Response for retraining status."""

    status: str = Field(..., description="Status of the job")


class LoadRequest(BaseModel):
    """Request for loading a model."""

    version: str = Field(..., description="Version of the model")


class LoadResponse(BaseModel):
    """Response for loading a model."""

    status: str = Field(..., description="Status of the job")


class Notification(BaseModel):
    """Notification for drift errors."""

    id: str = Field(..., description="ID of the notification")
    timestamp: datetime = Field(..., description="Timestamp of the notification")
    ml_task: MLTask = Field(..., description="ML task of the notification")
    message: str = Field(..., description="Message of the notification")


class SimulationStatus(BaseModel):
    """Status of the simulation."""

    state: SimulationState = Field(..., description="State of the simulation")
    current_sim_time: float = Field(..., description="Current simulation time")


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = Field(..., description="Status of the service")
    service: PlatformService = Field(..., description="Service")
