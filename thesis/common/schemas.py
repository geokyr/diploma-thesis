"""Shared Pydantic schemas for API models across all platform services."""

from datetime import datetime

from pydantic import BaseModel, Field

from thesis.common.enums import DriftState, MLTask, RetrainStatus, SimulationState
from thesis.common.service import PlatformService


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = Field(..., description="Status of the service")
    service: PlatformService = Field(..., description="Service")


class DriftInfo(BaseModel):
    """Complete drift info for a task held in the snapshot."""

    state: DriftState = Field(..., description="Drift state")
    start_timestamp: int | None = Field(None, description="Start timestamp of current state")
    collecting: bool = Field(..., description="Whether we are in collection window")
    job_id: str | None = Field(None, description="Current retrain job id, if any")


class SimulationSnapshot(BaseModel):
    """Snapshot of the simulation."""

    state: SimulationState = Field(..., description="State of the simulation")
    clock: int = Field(..., description="Current simulation clock time")
    drift_info: dict[MLTask, DriftInfo] = Field(..., description="Drift info per ML task")


class PredictionBatchRequest(BaseModel):
    """Request for batch predictions."""

    start_timestamp: int = Field(..., description="Window start timestamp for the predictions")
    end_timestamp: int = Field(..., description="Window end timestamp for the predictions")


class ErrorPoint(BaseModel):
    """Error point for predictions."""

    timestamp: int = Field(..., description="Timestamp of the error")
    error: float = Field(..., description="Error value")


class PredictionBatchResponse(BaseModel):
    """Response for batch predictions."""

    error_points: list[ErrorPoint] = Field(..., description="List of error points")
    mae: float | None = Field(None, description="Mean absolute error")


class MetricPoint(BaseModel):
    """Metric point for metrics."""

    timestamp: int = Field(..., description="Timestamp of the metric")
    mae: float | None = Field(None, description="Mean absolute error")


class MetricsRequest(BaseModel):
    """Request for metrics."""

    ml_task: MLTask = Field(..., description="ML task")


class MetricsResponse(BaseModel):
    """Response for metrics."""

    metric_points: list[MetricPoint] = Field(..., description="List of metric points")


class DriftErrorsRequest(BaseModel):
    """Request for drift errors."""

    ml_task: MLTask = Field(..., description="ML task")
    error_points: list[ErrorPoint] = Field(..., description="List of error points")


class DriftErrorsResponse(BaseModel):
    """Response for drift errors."""

    state: DriftState = Field(..., description="Drift state")
    start_timestamp: int = Field(..., description="Start timestamp of the state")


class DriftSetRequest(BaseModel):
    """Request to set drift state."""

    ml_task: MLTask = Field(..., description="ML task")
    state: DriftState = Field(..., description="Drift state to set")
    start_timestamp: int = Field(..., description="Start timestamp for the state")


class RetrainRequest(BaseModel):
    """Request for retraining."""

    start_timestamp: int = Field(..., description="Window start timestamp for retraining")
    end_timestamp: int = Field(..., description="Window end timestamp for retraining")


class RetrainResponse(BaseModel):
    """Response for retraining."""

    job_id: str = Field(..., description="Job ID")


class RetrainStatusRequest(BaseModel):
    """Request for retraining status."""

    job_id: str = Field(..., description="Job ID")


class RetrainStatusResponse(BaseModel):
    """Response for retraining status."""

    status: RetrainStatus = Field(..., description="Status of the job")
    post_adaptation_errors: list[float] | None = Field(
        None, description="Post-adaptation errors from retrained model (only available when status is COMPLETED)"
    )


class RecalibrateRequest(BaseModel):
    """Request for recalibrating drift detectors."""

    ml_task: MLTask = Field(..., description="ML task")
    post_adaptation_errors: list[float] = Field(..., description="Post-adaptation errors from retrained model")


class RecalibrateResponse(BaseModel):
    """Response for recalibrating drift detectors."""

    success: bool = Field(..., description="Whether the recalibration was successful")


class Notification(BaseModel):
    """Notification for drift errors."""

    id: str = Field(..., description="ID of the notification")
    timestamp: datetime = Field(..., description="Timestamp of the notification")
    ml_task: MLTask = Field(..., description="ML task of the notification")
    message: str = Field(..., description="Message of the notification")


class PredictionSingleRequest(BaseModel):
    """Request for single prediction."""

    source_latitude: float = Field(..., description="Source latitude")
    source_longitude: float = Field(..., description="Source longitude")
    destination_latitude: float = Field(..., description="Destination latitude")
    destination_longitude: float = Field(..., description="Destination longitude")
    start_timestamp: int = Field(..., description="Trip start time")


class PredictionSingleResponse(BaseModel):
    """Response for single prediction."""

    prediction: float | None = Field(None, description="Predicted value")


class TripPredictionResponse(BaseModel):
    """Response for trip prediction."""

    predictions: dict[MLTask, PredictionSingleResponse] = Field(..., description="Predictions per ML task")
