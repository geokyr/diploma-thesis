"""Shared Pydantic schemas for API models across all platform services."""

from pydantic import BaseModel, Field

from thesis.common.enums import (
    DriftState,
    MLTask,
    NotificationLevel,
    PlatformService,
    ReportStatus,
    RetrainStatus,
    SimulationState,
)


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
    mae: float = Field(..., description="Mean absolute error")
    n_samples: int = Field(..., description="Number of samples in this metric point")


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


class RetrainRequest(BaseModel):
    """Request for retraining."""

    start_timestamp: int = Field(..., description="Window start timestamp for retraining")
    end_timestamp: int = Field(..., description="Window end timestamp for retraining")


class RetrainResponse(BaseModel):
    """Response for retraining."""

    job_id: str = Field(..., description="Job ID")


class RetrainStatusResponse(BaseModel):
    """Response for retraining status."""

    status: RetrainStatus = Field(..., description="Status of the job")


class RecalibrateRequest(BaseModel):
    """Request for recalibrating drift detectors."""

    ml_task: MLTask = Field(..., description="ML task")


class RecalibrateResponse(BaseModel):
    """Response for recalibrating drift detectors."""

    success: bool = Field(..., description="Whether the recalibration was successful")


class DriftResetRequest(BaseModel):
    """Request for resetting drift detection."""

    ml_tasks: list[MLTask] = Field(..., description="List of ML tasks to reset drift detection for")


class DriftResetResponse(BaseModel):
    """Response for resetting drift detection."""

    success: bool = Field(..., description="Whether the reset was successful")


class Notification(BaseModel):
    """Notification for simulation events."""

    timestamp: int = Field(..., description="Simulation timestamp of the notification")
    message: str = Field(..., description="Message of the notification")
    level: NotificationLevel = Field(..., description="Level of the notification")
    ml_task: MLTask | None = Field(None, description="ML task of the notification")


class NotificationFeed(BaseModel):
    """Feed of notifications from the simulation."""

    notifications: list[Notification] = Field(..., description="List of notifications")


class PredictionSingleRequest(BaseModel):
    """Request for single prediction."""

    start_timestamp: int = Field(..., description="Trip start time")
    source_x: float = Field(..., description="Source x coordinate")
    source_y: float = Field(..., description="Source y coordinate")
    destination_x: float = Field(..., description="Destination x coordinate")
    destination_y: float = Field(..., description="Destination y coordinate")
    distance: float = Field(..., description="Trip distance in meters")
    edges: list[str] = Field(..., description="List of edge IDs along the trip")
    minimum_x: float = Field(..., description="Minimum x coordinate along the route")
    maximum_x: float = Field(..., description="Maximum x coordinate along the route")
    minimum_y: float = Field(..., description="Minimum y coordinate along the route")
    maximum_y: float = Field(..., description="Maximum y coordinate along the route")


class PredictionSingleResponse(BaseModel):
    """Response for single prediction."""

    prediction: float | None = Field(None, description="Predicted value")


class TripPredictionResponse(BaseModel):
    """Response for trip prediction."""

    predictions: dict[MLTask, PredictionSingleResponse] = Field(..., description="Predictions per ML task")


class RoutePreviewRequest(BaseModel):
    """Request for route preview."""

    source_latitude: float = Field(..., description="Source latitude")
    source_longitude: float = Field(..., description="Source longitude")
    destination_latitude: float = Field(..., description="Destination latitude")
    destination_longitude: float = Field(..., description="Destination longitude")


class RoutePreviewResponse(BaseModel):
    """Response for route preview."""

    source_x: float = Field(..., description="Source x coordinate")
    source_y: float = Field(..., description="Source y coordinate")
    destination_x: float = Field(..., description="Destination x coordinate")
    destination_y: float = Field(..., description="Destination y coordinate")
    distance: float = Field(..., description="Trip distance in meters")
    edges: list[str] = Field(..., description="List of edge IDs along the trip")
    route: list[tuple[float, float]] = Field(..., description="Route polyline as list of (lat, lon) tuples")
    minimum_x: float = Field(..., description="Minimum x coordinate along the route")
    maximum_x: float = Field(..., description="Maximum x coordinate along the route")
    minimum_y: float = Field(..., description="Minimum y coordinate along the route")
    maximum_y: float = Field(..., description="Maximum y coordinate along the route")


class ReportGenerationRequest(BaseModel):
    """Request for report generation."""

    notifications: list[Notification] = Field(..., description="List of notifications")
    metrics: dict[MLTask, MetricsResponse] = Field(..., description="Metrics for all ML tasks")


class ReportGenerationResponse(BaseModel):
    """Response for report generation."""

    content: str = Field(..., description="Content of the report")


class ReportStatusResponse(BaseModel):
    """Response for report status check."""

    status: ReportStatus = Field(..., description="Status of the report generation")
