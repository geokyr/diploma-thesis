# Plan

## Overview
Building a drift detection and mitigation platform with 3 ML models (ETA, fuel consumption, stops prediction) running on streaming Athens traffic data with concept drift simulation (test → rain scenarios).

## Key Requirements & Constraints
- **Simple Architecture**: 3 components only (Backend, Predictors, Frontend)
- **No Over-engineering**: Filesystem storage, no database, minimal complexity
- **Existing Code Reuse**: Leverage `thesis/` package for data processing and ML
- **Docker Orchestration**: uv + group dependencies per service
- **Real-time Demo**: 20h simulation data compressed to 2-3 minutes timelapse
- **Performance**: Handle 55-60k trips, ~333 trips/second peak with batching
- **Python 3.12.11**: uv environment management

## Architecture Components

### 1. Three-Service Architecture
```
thesis/
├── backend/                 # FastAPI orchestrator + drift detection
│   ├── main.py              # FastAPI app with WebSocket
│   ├── simulation.py        # Clock & trip feeder from parquet
│   ├── drift.py             # Drift detection (colleague integration)
│   ├── state.py             # JSON-based state management
│   └── models.py            # Pydantic API models
├── predictors/              # Model serving services (3 instances)
│   ├── main.py              # FastAPI model server
│   ├── predictor.py         # joblib model loading & prediction
│   └── preprocessing.py     # thesis.eta.features wrapper
├── frontend/                # Dash dashboard
│   ├── main.py              # Dash app
│   ├── admin_tab.py         # Real-time metrics & notifications
│   └── user_tab.py          # Athens map interface
├── common/
├── eta/
├── simulation/
platform/                    # Volume-mounted data (at project root)
├── data/                    # test-fcd.parquet, rain-fcd.parquet
│   ├── test-fcd.parquet
│   └── rain-fcd.parquet
├── models/                  # Model registry with versioning
│   ├── eta/{stable/, retrained/}
│   ├── fuel/{stable/, retrained/}
│   └── stops/{stable/, retrained/}
├── state/                   # Persistent state files
│   ├── simulation.json      # Current time, dataset, status
│   ├── model-versions.json  # Active model versions
│   └── drift-status.json    # Per-model drift states
├── docker-compose.dev.yml   # Development docker-compose file
└── docker-compose.yml       # Production docker-compose file
```

## Integration with Existing Code

### Data Pipeline Reuse
```python
# Existing thesis/ imports used directly:
from thesis.common.data import load_fcd_dataset, generate_trips, preprocess_fcd_dataset
from thesis.eta.features import add_all_features
from thesis.common.config import config  # Athens bbox, parameters
from thesis.eta.models import ModelType, create_model
```

### Feature Engineering Pipeline
```python
# predictors/preprocessing.py
def preprocess_batch(trip_data: list) -> pd.DataFrame:
    """Convert raw trip data to features using existing pipeline"""
    df = pd.DataFrame(trip_data)
    df_features = add_all_features(df)  # Your existing feature engineering
    return df_features
```

### Configuration Integration
```python
# Use existing config.yaml values:
ATHENS_BBOX = config["network"]["bbox"]  # [23.725252771719436, 37.974745936977456, 23.752735758169127, 37.988290142332225]
ETA_CONFIG = config["eta"]              # min_duration, min_distance, etc.
FEATURES_CONFIG = config["features"]    # n_clusters, coordinate_scale, etc.
```

## Service Specifications

### Backend Service (Port 8000)
**Responsibilities:**
- Simulation clock: 150ms ticks (1min sim time)
- Trip feeder: Load parquet data in chronological order
- Error calculator: Compare predictions with ground truth
- Drift detection: Integrate colleague's detection logic
- State manager: Persist simulation state, drift status
- WebSocket broadcaster: Push real-time updates to frontend
- API endpoints: `/start`, `/pause`, `/status`, `/user-predict`

**Key APIs:**
```python
POST /start              # Start simulation
POST /pause              # Pause/resume simulation
GET /status              # Current simulation state
POST /user-predict       # Custom trip prediction
WebSocket /ws            # Real-time metrics stream
```

### Predictor Services (Ports 8001-8003)
**Responsibilities:**
- Model loading: Load joblib models with preprocessing pipelines
- Batch prediction: Handle trip batches with feature engineering
- Model retraining: Async retraining with progress tracking
- Hot swapping: Load new model versions without downtime

**Key APIs:**
```python
POST /predict            # Batch prediction with features
POST /retrain            # Start async retraining
GET /status              # Training progress
POST /load               # Swap to new model version
```

### Frontend Service (Port 8080)
**Responsibilities:**
- Admin tab: Real-time MAE graphs, drift notifications, status indicators
- User tab: Athens map with origin/destination selection
- WebSocket client: Receive real-time updates from backend
- Map interface: Folium map with Athens bounding box

## Environment & Dependencies

### Service-Specific pyproject.toml Files

#### Backend
```toml
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn>=0.24.0",
    "websockets>=12.0",
    "httpx>=0.25.2",
    "pandas>=2.1.0",
    "pydantic>=2.5.0",
    "numpy>=1.24.0"
]
```

#### Predictors
```toml
dependencies = [
    "fastapi>=0.104.1",
    "uvicorn>=0.24.0",
    "joblib>=1.3.2",
    "pandas>=2.1.0",
    "scikit-learn>=1.3.0",
    "lightgbm>=4.0.0",
    "numpy>=1.24.0"
]
```

#### Frontend
```toml
dependencies = [
    "dash>=2.14.0",
    "plotly>=5.17.0",
    "folium>=0.14.0",
    "websocket-client>=1.6.0",
    "requests>=2.31.0"
]
```

## Data Flow Architecture

### Simulation Loop (Backend)
```python
async def simulation_loop():
    """Main simulation orchestrator"""
    while simulation_active:
        # 1. Load next batch of trips (1 min simulation time)
        current_batch = load_trip_batch(current_sim_time, BATCH_SIZE_MINUTES)

        # 2. Send to all predictors in parallel
        predictions = await asyncio.gather(
            predict_eta(current_batch),
            predict_fuel(current_batch),
            predict_stops(current_batch)
        )

        # 3. Calculate errors against ground truth
        errors = calculate_errors(predictions, ground_truth_batch)

        # 4. Update drift detector with error stream
        drift_events = update_drift_detection(errors)

        # 5. Handle drift events (start collecting/retraining)
        await handle_drift_events(drift_events)

        # 6. Broadcast metrics to frontend
        await websocket_broadcast({
            "metrics": calculate_rolling_mae(errors),
            "notifications": drift_events,
            "sim_time": current_sim_time
        })

        # 7. Check dataset transition (test → rain)
        if should_transition_to_rain(current_sim_time):
            await notify_day_change()

        await asyncio.sleep(SIMULATION_TICK_MS / 1000)  # 150ms default
```

### Prediction Pipeline (Predictors)
```python
@app.post("/predict")
async def predict_batch(request: PredictRequest):
    """Batch prediction with feature engineering"""
    # Convert batch to DataFrame
    df_trips = pd.DataFrame(request.trips)

    # Apply existing feature engineering pipeline
    df_features = preprocess_batch(df_trips)  # Uses add_all_features()

    # Predict using loaded model
    predictions = current_model.predict(df_features)

    return {"predictions": predictions.tolist()}
```

## State Management Strategy

### Persistent State (JSON Files)
```python
@dataclass
class SimulationState:
    current_time: int = 0
    dataset: str = "test"  # "test" or "rain"
    active: bool = False
    speed_multiplier: int = 400  # 20h → 3min

    def save_checkpoint(self, path: Path):
        with open(path / "simulation.json", "w") as f:
            json.dump(asdict(self), f)

@dataclass
class DriftState:
    eta_status: str = "stable"      # stable|collecting|retraining|swapped
    fuel_status: str = "stable"
    stops_status: str = "stable"

    # Rolling error buffers for drift detection
    eta_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    fuel_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    stops_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
```

### Model Registry Structure
```
platform/models/
├── eta/
│   ├── stable/
│   │   ├── model.joblib       # LightGBM/XGBoost model
│   │   └── metadata.json      # version, training_data, performance
│   └── retrained/
│       ├── model.joblib       # Retrained on test+rain data
│       └── metadata.json      # retraining timestamp, performance
├── fuel/
│   ├── stable/
│   └── retrained/
└── stops/
    ├── stable/
    └── retrained/
```

## Configuration Management

### Environment Variables (Configurable but not user-facing)
```bash
# Simulation timing
TIMELAPSE_SPEED_MULTIPLIER=400    # 20h → 3min (400x speedup)
SIMULATION_TICK_MS=150            # Real-time between ticks
BATCH_SIZE_MINUTES=1              # Simulation time per batch

# Drift detection
COLLECTION_WINDOW_TRIPS=1000      # Trips to collect before retraining
DRIFT_DETECTION_SMOOTHING=10      # Rolling window for error smoothing

# Service URLs
ETA_PREDICTOR_URL=http://eta-predictor:8000
FUEL_PREDICTOR_URL=http://fuel-predictor:8000
STOPS_PREDICTOR_URL=http://stops-predictor:8000
```

### Integration with thesis/common/config.yaml
```python
# Load existing configuration
from thesis.common.config import config

# Athens map bounding box
ATHENS_BBOX = config["network"]["bbox"]
# [23.725252771719436, 37.974745936977456, 23.752735758169127, 37.988290142332225]

# Feature engineering parameters
FEATURES_CONFIG = config["features"]
ETA_CONFIG = config["eta"]
```

## Frontend Design Specifications

### Admin Tab - Real-time Metrics Dashboard
- **3 Real-time MAE graphs**: One per model (ETA, fuel, stops)
- **Metrics display**: Current MAE values with timestamps
- **Drift status indicators**: Color-coded per model (green=stable, yellow=collecting, blue=retraining, red=drift)
- **Notification feed**: Timestamped drift events, model swaps, day transitions
- **Simulation controls**: Start/pause buttons, speed indicator
- **Update frequency**: 1Hz via WebSocket

### User Tab - Athens Map Interface
- **Interactive map**: Folium map centered on Athens bbox
- **Click handlers**: First click = source, second click = destination
- **Trip prediction**: Send coordinates to backend, display ETA/fuel/stops
- **Weather indicator**: Visual rain effect when rain dataset active
- **Drift badges**: Show which models are currently adapted
- **Current time display**: Show simulation timestamp

### Map Configuration
```python
def create_athens_map():
    bbox = config["network"]["bbox"]
    center_lat = (bbox[1] + bbox[3]) / 2  # 37.981518
    center_lon = (bbox[0] + bbox[2]) / 2  # 23.738994

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add simulation bounding box
    folium.Rectangle(
        bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
        color="red", fillOpacity=0.1, popup="Simulation Area"
    ).add_to(m)

    return m
```

## MVP Implementation Phases

### Phase 1: Core Foundation (3-4 days)
**Goal**: Working end-to-end pipeline with basic simulation

**Deliverables:**
- ✅ Backend with simulation clock reading test-fcd.parquet
- ✅ Single ETA predictor service with thesis.eta.features integration
- ✅ Basic HTTP communication (polling, no WebSocket yet)
- ✅ Docker Compose orchestration working
- ✅ Simple frontend displaying metrics via HTTP polling
- ✅ JSON-based state persistence

**Key Features:**
- Load parquet data chronologically in batches
- ETA model serving with feature engineering
- Basic error calculation and logging
- Manual start/pause controls
- Development environment setup

**Success Criteria:**
- `docker-compose up` works end-to-end
- ETA predictions flowing through pipeline
- Basic metrics visible in frontend
- State persists across restarts

### Phase 2: Full Model Pipeline (4-5 days)
**Goal**: All models working with real-time frontend

**Deliverables:**
- ✅ All 3 predictor services (ETA, fuel, stops)
- ✅ Parallel prediction calls with error aggregation
- ✅ Admin dashboard with 3 real-time MAE graphs
- ✅ WebSocket implementation for smooth updates
- ✅ Test→rain dataset transition with notifications
- ✅ Rolling metrics calculation (1Hz updates)

**Key Features:**
- Async prediction calls to all services
- MAE calculation per model per time window
- Dash frontend with live Plotly graphs
- Day transition notifications
- Error logging and debugging

**Success Criteria:**
- All 3 models predicting simultaneously
- Real-time graphs updating smoothly
- Dataset transition working correctly
- Performance acceptable (no lag/blocking)

### Phase 3: Drift Integration (5-6 days)
**Goal**: Complete drift detection and mitigation workflow

**Deliverables:**
- ✅ Drift detection integrated in backend (colleague's algorithm)
- ✅ Complete retraining workflow with async progress
- ✅ Model hot-swapping without service downtime
- ✅ Drift state visualization and notifications
- ✅ Model versioning (stable/retrained) management

**Key Features:**
- HTTP endpoint design for colleague's drift service
- Background retraining tasks with progress tracking
- Model registry with version management
- Visual drift status indicators (colors/badges)
- Notification system for all drift events

**Success Criteria:**
- Drift detection triggers correctly on rain data
- Retraining completes and model swaps
- Error rates improve after model swap
- All drift states tracked and visualized
- Colleague can integrate drift service easily

### Phase 4: User Experience (3-4 days)
**Goal**: Complete user interface and final polish

**Deliverables:**
- ✅ User tab with interactive Athens map
- ✅ Custom trip prediction interface
- ✅ Rain effect visualization on map
- ✅ Performance optimization and monitoring
- ✅ Final testing and documentation

**Key Features:**
- Folium map with Athens bounding box
- Click handlers for origin/destination selection
- Trip prediction at current simulation time
- Weather/drift status badges on map
- Performance monitoring and optimization

**Success Criteria:**
- Map interface works smoothly
- Custom predictions return reasonable results
- Rain effects visible and appropriate
- System performs well under full load
- Ready for demonstration

## Performance & Risk Mitigation

### Performance Monitoring
```python
import time
from functools import wraps

def time_function(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper

# Apply to critical functions
@time_function
async def feature_calculation_batch(trips: list) -> pd.DataFrame:
    # Monitor add_all_features() performance

@time_function
async def model_prediction_batch(features: pd.DataFrame) -> np.ndarray:
    # Monitor model inference time
```

### Graceful Degradation Strategies
1. **Slow feature calculation**: Reduce batch size, consider pre-computation
2. **Model service timeouts**: Return cached predictions + warning notification
3. **Memory issues**: Implement sliding window cleanup of old error buffers
4. **Drift detection lag**: Queue error streams if processing falls behind
5. **WebSocket connection issues**: Fall back to HTTP polling

### Configurable Performance Tuning
```python
# All timing adjustable via environment variables
SIMULATION_TICK_MS = int(os.getenv("SIMULATION_TICK_MS", 150))
BATCH_SIZE_MINUTES = int(os.getenv("BATCH_SIZE_MINUTES", 1))
COLLECTION_WINDOW_TRIPS = int(os.getenv("COLLECTION_WINDOW_TRIPS", 1000))
MAX_ERROR_BUFFER_SIZE = int(os.getenv("MAX_ERROR_BUFFER_SIZE", 10000))
```

## Colleague Integration Points

### Drift Service API Design
```python
# HTTP endpoint for colleague's drift detection service
@app.post("/drift/update")
async def update_drift_detector(request: ErrorStreamUpdate):
    """Receive error stream batch from backend"""
    for model_type, errors in request.error_batches.items():
        drift_result = await drift_detector.update(model_type, errors)
        if drift_result.drift_detected:
            await notify_backend_drift_event(model_type, drift_result)

    return {"status": "processed", "timestamp": time.time()}
```

### Frontend Integration Points
```python
# Clear API contracts for frontend colleague
class MetricsUpdate(BaseModel):
    timestamp: float
    eta_mae: float
    fuel_mae: float
    stops_mae: float
    drift_status: dict[str, str]  # model -> status

class NotificationEvent(BaseModel):
    timestamp: float
    type: str  # "drift_detected", "retrain_started", "model_swapped", "day_transition"
    model: str | None
    message: str
```

## Future Extension Points

### Scalability Considerations
- **Horizontal scaling**: Stateless services can run multiple replicas
- **Database migration**: Clear data models for future DB integration
- **Message queues**: Replace HTTP with async queues for higher throughput
- **Monitoring**: Prometheus/Grafana integration points identified

### Feature Extensions
- **Multiple drift scenarios**: Framework supports additional datasets
- **A/B testing**: Model comparison infrastructure in place
- **Real-time data**: Interface designed for live data streams
- **Advanced visualizations**: Modular frontend components

## Success Criteria & Deliverables

### Phase 1 Success Criteria
- [ ] End-to-end pipeline working with ETA predictions
- [ ] Docker orchestration functional
- [ ] Basic metrics display
- [ ] State persistence working

### Phase 2 Success Criteria
- [ ] All 3 models predicting simultaneously
- [ ] Real-time graphs updating at 1Hz
- [ ] Test→rain transition working
- [ ] Performance adequate for demo

### Phase 3 Success Criteria
- [ ] Drift detection working on rain data
- [ ] Retraining and model swapping functional
- [ ] All drift states visualized
- [ ] Colleague integration points ready

### Phase 4 Success Criteria
- [ ] Interactive map working smoothly
- [ ] Custom trip predictions accurate
- [ ] System ready for demonstration
- [ ] Performance optimized

This plan provides a solid, implementable foundation that leverages your existing work while keeping complexity minimal. Ready to proceed with Phase 1 implementation.
