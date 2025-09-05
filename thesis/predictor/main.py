"""
Model serving FastAPI service for ETA, fuel, and stops prediction.
"""

import time
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from thesis.common.logger import setup_logger
from thesis.common.service import PlatformServiceConfig
from thesis.predictor.predictor import ModelPredictor

config = PlatformServiceConfig()
logger = setup_logger(config.service, config.logs_dir)


# API Models
class TripData(BaseModel):
    source_x: float
    source_y: float
    destination_x: float
    destination_y: float
    time_start: int
    distance: float
    duration: float = None  # Optional ground truth


class PredictRequest(BaseModel):
    trips: List[TripData]
    task_type: str


class PredictResponse(BaseModel):
    predictions: List[float]
    task_type: str
    timestamp: float
    count: int


class StatusResponse(BaseModel):
    task_type: str
    model_loaded: bool
    current_version: str
    model_metadata: dict
    timestamp: float


class LoadModelRequest(BaseModel):
    version: str = "stable"  # "stable" or "retrained"


class LoadModelResponse(BaseModel):
    success: bool
    message: str
    version: str
    timestamp: float


# Initialize FastAPI app
app = FastAPI(
    title=f"Model Predictor - {config.task.upper()}",
    description=f"Model serving service for {config.task} prediction",
    version="0.1.0",
)

# Global predictor instance
predictor: ModelPredictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize predictor service on startup."""
    global predictor

    logger.info(f"Starting {config.task} predictor service...")

    # Ensure models directory exists
    config.models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize predictor
    predictor = ModelPredictor(config.task, config.models_dir)

    # Try to load default model
    success = await predictor.load_model("stable")
    if not success:
        logger.warning(f"Could not load default model for {config.task}")

        # For Phase 1 testing, try to find any available model in outputs
        fallback_paths = [
            Path("outputs/final_model/models/LGBMRegressor.joblib"),
            Path("outputs/all_features_optimized/models/LGBMRegressor.joblib"),
            Path("outputs/all_features/models/LGBMRegressor.joblib"),
        ]

        for fallback_path in fallback_paths:
            if fallback_path.exists():
                logger.info(f"Found fallback model at {fallback_path}")
                try:
                    import shutil

                    # Copy to platform models location
                    stable_dir = config.models_dir / config.task / "stable"
                    stable_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(fallback_path, stable_dir / "model.joblib")

                    # Try loading again
                    success = await predictor.load_model("stable")
                    if success:
                        logger.info(f"Successfully loaded fallback model for {config.task}")
                        break
                except Exception as e:
                    logger.error(f"Failed to copy fallback model: {e}")

        if not success:
            logger.warning(f"No model available for {config.task} - service will return errors")

    logger.info(f"{config.task} predictor service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info(f"Shutting down {config.task} predictor service")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "task_type": config.task, "timestamp": time.time()}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Main prediction endpoint."""
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")

        if predictor.current_model is None:
            raise HTTPException(status_code=503, detail=f"No model loaded for {config.task}")

        # Validate request
        if not request.trips:
            raise HTTPException(status_code=400, detail="No trips provided")

        if request.task_type != config.task:
            raise HTTPException(
                status_code=400, detail=f"Task type mismatch: expected {config.task}, got {request.task_type}"
            )

        # Convert trips to dictionaries
        trip_data = [trip.dict() for trip in request.trips]

        # Get predictions
        predictions = await predictor.predict_batch(trip_data)

        return PredictResponse(
            predictions=predictions, task_type=config.task, timestamp=time.time(), count=len(predictions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get predictor service status."""
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")

        status_data = predictor.get_status()

        return StatusResponse(
            task_type=status_data["task_type"],
            model_loaded=status_data["model_loaded"],
            current_version=status_data["current_version"],
            model_metadata=status_data["model_metadata"],
            timestamp=time.time(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load", response_model=LoadModelResponse)
async def load_model(request: LoadModelRequest):
    """Load a specific model version."""
    try:
        if not predictor:
            raise HTTPException(status_code=500, detail="Predictor not initialized")

        if request.version not in ["stable", "retrained"]:
            raise HTTPException(status_code=400, detail="Version must be 'stable' or 'retrained'")

        success = await predictor.load_model(request.version)

        if success:
            message = f"Successfully loaded {request.version} model for {config.task}"
            logger.info(message)
        else:
            message = f"Failed to load {request.version} model for {config.task}"
            logger.error(message)

        return LoadModelResponse(
            success=success,
            message=message,
            version=request.version if success else predictor.current_version,
            timestamp=time.time(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain_model():
    """Placeholder for retraining endpoint (Phase 3)."""
    logger.info(f"Retrain requested for {config.task} (not implemented yet)")
    return {
        "message": f"Retraining for {config.task} will be implemented in Phase 3",
        "task_type": config.task,
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
