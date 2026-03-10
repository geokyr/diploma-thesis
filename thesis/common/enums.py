"""Shared enums for the project."""

from enum import StrEnum


class SimulationScenario(StrEnum):
    """
    Simulation Scenarios.

    Attributes:
        TRAIN: Train data on base network.
        TEST: Test data on base network.
        RAIN: Retrain/test data on rain network.
    """

    TRAIN = "train"
    TEST = "test"
    RAIN = "rain"


class ETAEvaluation(StrEnum):
    """
    ETA Evaluations.

    Attributes:
        RESEARCH: Evaluate via cross-validation on training data.
        STABLE: Evaluate on a clean, held-out test set.
        DRIFT: Evaluate on drift-affected data to quantify degradation.
        RETRAIN: Evaluate after retraining on drift-affected data to assess recovery.
    """

    RESEARCH = "research"
    STABLE = "stable"
    DRIFT = "drift"
    RETRAIN = "retrain"


class MLTask(StrEnum):
    """
    ML tasks.

    Attributes:
        ETA: Estimated time of arrival prediction task.
        FUEL: Fuel consumption prediction task.
        STOPS: Number of stops prediction task.
    """

    ETA = "eta"
    FUEL = "fuel"
    STOPS = "stops"


class ModelType(StrEnum):
    """
    Model Types.

    Attributes:
        LINEAR_REGRESSION: Linear Regression.
        XGBOOST_REGRESSOR: XGBoost Regressor.
        LIGHTGBM_REGRESSOR: LightGBM Regressor.
        CATBOOST_REGRESSOR: CatBoost Regressor.
    """

    LINEAR_REGRESSION = "LinearRegression"
    XGBOOST_REGRESSOR = "XGBRegressor"
    LIGHTGBM_REGRESSOR = "LGBMRegressor"
    CATBOOST_REGRESSOR = "CatBoostRegressor"


class FeatureGroup(StrEnum):
    """
    Feature groups.

    Attributes:
        TEMPORAL: Temporal features.
        SPATIAL: Spatial features.
        FOURIER: Fourier features.
        CELL: Cell features.
        CLUSTER: Cluster features.
        PCA: PCA features.
    """

    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    FOURIER = "fourier"
    CELL = "cell"
    CLUSTER = "cluster"
    PCA = "pca"


class DriftState(StrEnum):
    """
    Drift states.

    Attributes:
        CALIBRATING: Calibration state.
        STABLE: Stable state.
        DRIFTED: Drifted state.
        RETRAINING: Retraining state.
    """

    CALIBRATING = "calibrating"
    STABLE = "stable"
    DRIFTED = "drifted"
    RETRAINING = "retraining"


class SimulationState(StrEnum):
    """
    Simulation states.

    Attributes:
        READY: Ready state.
        RUNNING: Running state.
        PAUSED: Paused state.
        COMPLETED: Completed state.
    """

    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


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
        SUMMARIZER: Summarizer service.
    """

    BACKEND = "backend"
    PREDICTOR_ETA = "predictor-eta"
    PREDICTOR_FUEL = "predictor-fuel"
    PREDICTOR_STOPS = "predictor-stops"
    FRONTEND = "frontend"
    DRIFT = "drift"
    SUMMARIZER = "summarizer"


class PlatformServiceStatus(StrEnum):
    """
    Platform service statuses.

    Attributes:
        HEALTHY: Healthy status.
        UNHEALTHY: Unhealthy status.
    """

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class RetrainStatus(StrEnum):
    """
    Retrain statuses.

    Attributes:
        COMPLETED: Completed status.
        FAILED: Failed status.
        RUNNING: Running status.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


class DriftDetectorType(StrEnum):
    """
    Drift detector types.

    Attributes:
        ADWIN: Adaptive Windowing detector.
        PAGE_HINKLEY: Page-Hinkley test detector.
        KSWIN: Kolmogorov-Smirnov Windowing detector.
        SPC: Statistical Process Control detector.
    """

    ADWIN = "adwin"
    PAGE_HINKLEY = "page_hinkley"
    KSWIN = "kswin"
    SPC = "spc"


class NotificationLevel(StrEnum):
    """
    Notification levels.

    Attributes:
        INFO: Info level.
        SUCCESS: Success level.
        WARNING: Warning level.
        DANGER: Danger level.
    """

    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"


class ReportStatus(StrEnum):
    """
    Report generation statuses.

    Attributes:
        NOT_STARTED: Not started status.
        GENERATING: Generating status.
        READY: Ready status.
        FAILED: Failed status.
    """

    NOT_STARTED = "not_started"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
