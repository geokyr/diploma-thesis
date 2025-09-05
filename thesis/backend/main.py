"""
Main FastAPI backend service for drift detection platform.
"""

import asyncio
import time
from typing import List, Optional

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from thesis.backend.models import (
    MetricsUpdate,
    NotificationEvent,
    PredictRequest,
    SimulationStatus,
    TripData,
    UserPredictRequest,
    UserPredictResponse,
)
from thesis.backend.simulation import SimulationEngine
from thesis.backend.state import DriftState, ModelVersions
from thesis.common.logger import setup_logger
from thesis.common.service import PlatformServiceConfig

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)

# Initialize FastAPI app
app = FastAPI(
    title="Drift Detection Platform Backend",
    description="Backend orchestrator for ML drift detection and mitigation",
    version="0.1.0",
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
simulation_engine: Optional[SimulationEngine] = None
drift_state = DriftState()
model_versions = ModelVersions()
connected_websockets: List[WebSocket] = []

# HTTP client for predictor services
http_client = httpx.AsyncClient(timeout=30.0)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global simulation_engine, model_versions

    logger.info("Starting drift detection platform backend...")

    # Ensure platform directories exist
    for path in [config.data_dir, config.state_dir, config.models_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Initialize simulation engine
    simulation_engine = SimulationEngine(config.data_dir, config.state_dir)
    simulation_engine.on_metrics_update = handle_metrics_update
    simulation_engine.on_notification = handle_notification

    # Load model versions
    model_versions = ModelVersions.load(config.state_dir)

    logger.info("Backend startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down backend...")

    if simulation_engine:
        await simulation_engine.shutdown()

    await http_client.aclose()
    logger.info("Backend shutdown complete")


async def handle_metrics_update(trips: List[TripData], ground_truth: List[float]):
    """Handle new trip batch for metrics calculation."""
    try:
        # Get predictions from all models in parallel
        eta_predictions, fuel_predictions, stops_predictions = await asyncio.gather(
            predict_batch(trips, "eta"),
            predict_batch(trips, "fuel"),
            predict_batch(trips, "stops"),
            return_exceptions=True,
        )

        # Calculate errors (handle prediction failures gracefully)
        eta_errors = []
        fuel_errors = []
        stops_errors = []

        if not isinstance(eta_predictions, Exception) and eta_predictions:
            eta_errors = [abs(pred - truth) for pred, truth in zip(eta_predictions, ground_truth)]

        if not isinstance(fuel_predictions, Exception) and fuel_predictions:
            fuel_errors = [abs(pred - truth) for pred, truth in zip(fuel_predictions, ground_truth)]

        if not isinstance(stops_predictions, Exception) and stops_predictions:
            stops_errors = [abs(pred - truth) for pred, truth in zip(stops_predictions, ground_truth)]

        # Update drift state with errors
        if eta_errors:
            for error in eta_errors:
                drift_state.add_errors(eta_error=error)
        if fuel_errors:
            for error in fuel_errors:
                drift_state.add_errors(fuel_error=error)
        if stops_errors:
            for error in stops_errors:
                drift_state.add_errors(stops_error=error)

        # Get current MAE values
        current_mae = drift_state.get_current_mae()

        # Create metrics update
        metrics_update = MetricsUpdate(
            timestamp=time.time(),
            simulation_time=simulation_engine.state.current_time,
            dataset=simulation_engine.state.dataset,
            eta_mae=current_mae["eta"],
            fuel_mae=current_mae["fuel"],
            stops_mae=current_mae["stops"],
            active=simulation_engine.state.active,
        )

        # Broadcast to connected WebSockets
        await broadcast_metrics(metrics_update)

        # Save drift status periodically
        if len(drift_state.eta_errors) % 100 == 0:  # Every 100 batches
            drift_state.save_status(config.state_dir)

    except Exception as e:
        logger.error(f"Error in metrics update: {e}")


async def handle_notification(notification_data: dict):
    """Handle notification events."""
    notification = NotificationEvent(
        timestamp=notification_data.get("timestamp", time.time()),
        type=notification_data["type"],
        model=notification_data.get("model"),
        message=notification_data["message"],
    )

    await broadcast_notification(notification)


async def predict_batch(trips: List[TripData], task_type: str) -> List[float]:
    """Send batch prediction request to appropriate predictor service."""
    try:
        # Select service URL
        service_urls = {
            "eta": config.predictor_eta_url,
            "fuel": config.predictor_fuel_url,
            "stops": config.predictor_stops_url,
        }

        service_url = service_urls.get(task_type)
        if not service_url:
            raise ValueError(f"Unknown task type: {task_type}")

        # Prepare request
        request_data = PredictRequest(trips=[trip.dict() for trip in trips], task_type=task_type)

        # Send prediction request
        response = await http_client.post(f"{service_url}/predict", json=request_data.dict())
        response.raise_for_status()

        result = response.json()
        return result["predictions"]

    except Exception as e:
        logger.error(f"Prediction failed for {task_type}: {e}")
        raise


async def broadcast_metrics(metrics: MetricsUpdate):
    """Broadcast metrics to all connected WebSocket clients."""
    if not connected_websockets:
        return

    message = metrics.dict()
    disconnected = []

    for websocket in connected_websockets:
        try:
            await websocket.send_json({"type": "metrics", "data": message})
        except Exception as e:
            logger.warning(f"Failed to send metrics to WebSocket: {e}")
            disconnected.append(websocket)

    # Remove disconnected WebSockets
    for ws in disconnected:
        connected_websockets.remove(ws)


async def broadcast_notification(notification: NotificationEvent):
    """Broadcast notification to all connected WebSocket clients."""
    if not connected_websockets:
        return

    message = notification.dict()
    disconnected = []

    for websocket in connected_websockets:
        try:
            await websocket.send_json({"type": "notification", "data": message})
        except Exception as e:
            logger.warning(f"Failed to send notification to WebSocket: {e}")
            disconnected.append(websocket)

    # Remove disconnected WebSockets
    for ws in disconnected:
        connected_websockets.remove(ws)


# API Routes


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/start")
async def start_simulation():
    """Start the simulation."""
    try:
        if not simulation_engine:
            raise HTTPException(status_code=500, detail="Simulation engine not initialized")

        await simulation_engine.start_simulation()
        return {"message": "Simulation started", "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Failed to start simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pause")
async def pause_simulation():
    """Pause/resume the simulation."""
    try:
        if not simulation_engine:
            raise HTTPException(status_code=500, detail="Simulation engine not initialized")

        await simulation_engine.pause_simulation()
        status = "resumed" if simulation_engine.state.active else "paused"
        return {"message": f"Simulation {status}", "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Failed to pause simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=SimulationStatus)
async def get_simulation_status():
    """Get current simulation status."""
    try:
        if not simulation_engine:
            raise HTTPException(status_code=500, detail="Simulation engine not initialized")

        status_data = simulation_engine.get_status()
        return SimulationStatus(**status_data)

    except Exception as e:
        logger.error(f"Failed to get simulation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user-predict", response_model=UserPredictResponse)
async def user_predict(request: UserPredictRequest):
    """Handle custom user prediction request."""
    try:
        # Calculate distance (simple Euclidean for now)
        distance = np.sqrt(
            (request.destination_x - request.source_x) ** 2 + (request.destination_y - request.source_y) ** 2
        )

        # Create trip data
        trip = TripData(
            source_x=request.source_x,
            source_y=request.source_y,
            destination_x=request.destination_x,
            destination_y=request.destination_y,
            time_start=request.current_sim_time,
            distance=distance,
        )

        # Get predictions from all models
        eta_pred, fuel_pred, stops_pred = await asyncio.gather(
            predict_batch([trip], "eta"),
            predict_batch([trip], "fuel"),
            predict_batch([trip], "stops"),
            return_exceptions=True,
        )

        # Handle prediction failures
        eta_prediction = eta_pred[0] if not isinstance(eta_pred, Exception) and eta_pred else 0.0
        fuel_prediction = fuel_pred[0] if not isinstance(fuel_pred, Exception) and fuel_pred else 0.0
        stops_prediction = stops_pred[0] if not isinstance(stops_pred, Exception) and stops_pred else 0.0

        return UserPredictResponse(
            eta_prediction=eta_prediction,
            fuel_prediction=fuel_prediction,
            stops_prediction=stops_prediction,
            distance=distance,
            simulation_time=request.current_sim_time,
            drift_status=drift_state.get_drift_status(),
        )

    except Exception as e:
        logger.error(f"User prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    connected_websockets.append(websocket)

    logger.info(f"WebSocket connected. Total connections: {len(connected_websockets)}")

    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
        logger.info(f"WebSocket removed. Total connections: {len(connected_websockets)}")


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
