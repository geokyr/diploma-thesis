"""
Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class TripData(BaseModel):
    """Individual trip data for prediction."""

    source_x: float
    source_y: float
    destination_x: float
    destination_y: float
    time_start: int
    distance: float
    duration: Optional[int] = None  # Ground truth for error calculation


class PredictRequest(BaseModel):
    """Request for batch prediction."""

    trips: List[TripData]
    task_type: str  # "eta", "fuel", "stops"


class PredictResponse(BaseModel):
    """Response from prediction service."""

    predictions: List[float]
    task_type: str
    timestamp: float


class MetricsUpdate(BaseModel):
    """Real-time metrics update for frontend."""

    timestamp: float
    simulation_time: int
    dataset: str  # "test" or "rain"
    eta_mae: Optional[float] = None
    fuel_mae: Optional[float] = None
    stops_mae: Optional[float] = None
    active: bool


class NotificationEvent(BaseModel):
    """Notification event for frontend."""

    timestamp: float
    type: str  # "drift_detected", "retrain_started", "model_swapped", "day_transition"
    model: Optional[str] = None  # "eta", "fuel", "stops"
    message: str


class SimulationStatus(BaseModel):
    """Current simulation status."""

    active: bool
    current_time: int
    dataset: str
    speed_multiplier: int
    progress_percent: float


class UserPredictRequest(BaseModel):
    """Request for custom user prediction."""

    source_x: float
    source_y: float
    destination_x: float
    destination_y: float
    current_sim_time: int


class UserPredictResponse(BaseModel):
    """Response for custom user prediction."""

    eta_prediction: float
    fuel_prediction: float
    stops_prediction: float
    distance: float
    simulation_time: int
    drift_status: Dict[str, str]  # model -> status
