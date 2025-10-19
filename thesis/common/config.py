"""Configuration management module for the project."""

import os
from dataclasses import dataclass
from pathlib import Path
from sysconfig import get_paths

import yaml

PURELIB_PATH = Path(get_paths()["purelib"])
SUMO_HOME = PURELIB_PATH / "sumo"
os.environ["SUMO_HOME"] = str(SUMO_HOME)

SUMO_BIN = SUMO_HOME / "bin"
SUMO_TOOLS = SUMO_HOME / "tools"
PATHS = [os.environ.get("PATH", ""), str(SUMO_BIN), str(SUMO_TOOLS)]
os.environ["PATH"] = os.pathsep.join(PATHS)

OSM_GET = SUMO_TOOLS / "osmGet.py"
OSM_BUILD = SUMO_TOOLS / "osmBuild.py"
RANDOM_TRIPS = SUMO_TOOLS / "randomTrips.py"
NETCONVERT_TYPEMAP = SUMO_HOME / "data" / "typemap" / "osmNetconvert.typ.xml"
POLYCONVERT_TYPEMAP = SUMO_HOME / "data" / "typemap" / "osmPolyconvert.typ.xml"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    max_file_size: int
    backup_count: int


@dataclass(frozen=True, slots=True)
class DirnameConfig:
    simulation: str
    outputs: str
    data: str
    logs: str
    plots: str
    models: str
    results: str
    appdata: str
    misc: str
    common: str
    stable_models: str


@dataclass(frozen=True, slots=True)
class FilenameConfig:
    osm_data: str
    gui_settings: str
    poly: str
    network_base: str
    network_rain: str
    routes_temp: str
    logs: str
    results: str
    research_results: str
    tuning_results: str
    feature_calibrator: str
    trips_parquet: str
    model: str
    metadata: str
    feature_selection_results: str
    simulation_report: str


@dataclass(frozen=True, slots=True)
class SuffixConfig:
    trips: str
    sumocfg: str
    fcd_csv: str
    fcd_parquet: str


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    bbox: list[float]
    road_types: str
    vehicle_classes: str
    netconvert_options: str
    polyconvert_options: str
    friction: float
    view_settings: str


@dataclass(frozen=True, slots=True)
class SeedConfig:
    default: int
    train: int
    test: int
    rain: int


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    traffic_generation_periods_mean: float
    traffic_generation_periods_std: float
    traffic_generation_periods: list[float]
    tls_actuated_jam_threshold: int
    device_rerouting_adaptation_steps: int
    device_rerouting_adaptation_interval: int
    fcd_output_attributes: str
    start_time: int
    end_time: int
    device_friction_probability: float


@dataclass(frozen=True, slots=True)
class ExternalConfig:
    zenodo_dataset_api_url: str


@dataclass(frozen=True, slots=True)
class EtaConfig:
    target_column: str
    chunk_size: int
    min_duration: int
    min_distance: int
    augmentation_rate: float
    min_trip_ratio: float
    num_retraining_trips: int
    n_bins: int
    n_splits: int
    n_training_samples: int
    shrink_factor: float


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    morning_ceiling: int
    noon_floor: int
    noon_ceiling: int
    afternoon_floor: int
    rush_hours: list[int]
    percentile_thresholds: list[int]
    num_freqs: int
    coordinate_scale: int
    cell: int
    n_clusters: int
    n_components: int
    correlation_threshold: float
    permutation_importance_n_repeats: int
    permutation_importance_scoring: str
    permutation_importance_n_jobs: int
    ranking_alpha: float


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    objective_xgboost: str
    objective_lightgbm: str
    loss_function_catboost: str
    verbose_lightgbm: int
    verbose_catboost: int
    enable_categorical: bool
    allow_writing_files: bool
    tree_method: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    n_trials: int
    n_focused_trials: int
    direction: str
    importance_type: str


@dataclass(frozen=True, slots=True)
class ServicesConfig:
    host: str
    environment: str
    loop: str
    http: str
    access_log: bool
    backend: int
    predictor_eta: int
    predictor_fuel: int
    predictor_stops: int
    drift: int
    frontend: int
    summarizer: int


@dataclass(frozen=True, slots=True)
class TimelapseConfig:
    speed_multiplier: int
    interval_seconds: int
    pause_poll_seconds: float
    client_timeout_seconds: int
    collect_seconds: int
    metrics_maxlen: int
    notifications_maxlen: int


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    default_version: str


@dataclass(frozen=True, slots=True)
class PredictorEtaConfig:
    source_x_column: str
    source_y_column: str
    destination_x_column: str
    destination_y_column: str
    time_start_column: str
    distance_column: str


@dataclass(frozen=True, slots=True)
class DriftConfig:
    consensus_threshold: int
    smoothing_window_samples: int
    grace_period_samples: int
    calibration_window_samples: int
    adwin_delta_candidates: list[float]
    page_hinkley_delta_candidates: list[float]
    page_hinkley_threshold_candidates: list[float]
    kswin_alpha_candidates: list[float]
    kswin_window_size_stat_size_configs: list[list[int]]
    spc_n_std_candidates: list[float]
    spc_min_std: float
    spc_consecutive_violations_required_multiplier_configs: list[list[int]]


@dataclass(frozen=True, slots=True)
class FeatureCategoriesConfig:
    original: list[str]
    temporal: list[str]
    spatial: list[str]
    fourier: list[str]
    cell: list[str]
    cluster: list[str]
    pca: list[str]


@dataclass(frozen=True, slots=True)
class FuelConfig:
    time_start_column: str
    target_column: str
    random_seed: int
    min_trip_points: int
    n_start_clusters: int
    n_end_clusters: int
    n_init: int
    start_hour_max: int


@dataclass(frozen=True, slots=True)
class StopsConfig:
    time_start_column: str
    target_column: str
    random_seed: int
    n_clusters: int
    n_init: int
    min_trip_records: int
    min_duration: int
    min_distance: int
    use_spatial_features: bool
    use_route_features: bool


@dataclass(frozen=True, slots=True)
class SummarizerConfig:
    max_retries: int
    retry_delay_seconds: float
    async_client_timeout_seconds: float
    openrouter_base_url: str
    openrouter_model: str
    system_prompt: str


@dataclass(frozen=True, slots=True)
class Config:
    logging: LoggingConfig
    dirname: DirnameConfig
    filename: FilenameConfig
    suffix: SuffixConfig
    network: NetworkConfig
    seed: SeedConfig
    simulation: SimulationConfig
    external: ExternalConfig
    eta: EtaConfig
    features: FeaturesConfig
    models: ModelsConfig
    services: ServicesConfig
    timelapse: TimelapseConfig
    predictor: PredictorConfig
    predictor_eta: PredictorEtaConfig
    drift: DriftConfig
    feature_categories: FeatureCategoriesConfig
    fuel: FuelConfig
    stops: StopsConfig
    summarizer: SummarizerConfig


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        return Config(
            logging=LoggingConfig(**data["logging"]),
            dirname=DirnameConfig(**data["dirname"]),
            filename=FilenameConfig(**data["filename"]),
            suffix=SuffixConfig(**data["suffix"]),
            network=NetworkConfig(**data["network"]),
            seed=SeedConfig(**data["seed"]),
            simulation=SimulationConfig(**data["simulation"]),
            external=ExternalConfig(**data["external"]),
            eta=EtaConfig(**data["eta"]),
            features=FeaturesConfig(**data["features"]),
            models=ModelsConfig(**data["models"]),
            services=ServicesConfig(**data["services"]),
            timelapse=TimelapseConfig(**data["timelapse"]),
            predictor=PredictorConfig(**data["predictor"]),
            predictor_eta=PredictorEtaConfig(**data["predictor_eta"]),
            drift=DriftConfig(**data["drift"]),
            feature_categories=FeatureCategoriesConfig(**data["feature_categories"]),
            fuel=FuelConfig(**data["fuel"]),
            stops=StopsConfig(**data["stops"]),
            summarizer=SummarizerConfig(**data["summarizer"]),
        )


CONFIG_PATH = Path(__file__).with_name("config.yaml")
CONFIG = load_config(CONFIG_PATH)

PROJECT_DIR = Path(__file__).parent.parent.parent
SIMULATION_DIR = PROJECT_DIR / CONFIG.dirname.simulation
OUTPUTS_DIR = PROJECT_DIR / CONFIG.dirname.outputs

MAX_FILE_SIZE = CONFIG.logging.max_file_size
BACKUP_COUNT = CONFIG.logging.backup_count

DATA_DIRNAME = CONFIG.dirname.data
LOGS_DIRNAME = CONFIG.dirname.logs
PLOTS_DIRNAME = CONFIG.dirname.plots
MODELS_DIRNAME = CONFIG.dirname.models
RESULTS_DIRNAME = CONFIG.dirname.results
APPDATA_DIRNAME = CONFIG.dirname.appdata
MISC_DIRNAME = CONFIG.dirname.misc
COMMON_DIRNAME = CONFIG.dirname.common
STABLE_MODELS_DIRNAME = CONFIG.dirname.stable_models

OSM_DATA_FILENAME = CONFIG.filename.osm_data
GUI_SETTINGS_FILENAME = CONFIG.filename.gui_settings
POLY_FILENAME = CONFIG.filename.poly
NETWORK_BASE_FILENAME = CONFIG.filename.network_base
NETWORK_RAIN_FILENAME = CONFIG.filename.network_rain
ROUTES_TEMP_FILENAME = CONFIG.filename.routes_temp
LOGS_FILENAME = CONFIG.filename.logs
RESULTS_FILENAME = CONFIG.filename.results
RESEARCH_RESULTS_FILENAME = CONFIG.filename.research_results
TUNING_RESULTS_FILENAME = CONFIG.filename.tuning_results
FEATURE_CALIBRATOR_FILENAME = CONFIG.filename.feature_calibrator
TRIPS_PARQUET_FILENAME = CONFIG.filename.trips_parquet
MODEL_FILENAME = CONFIG.filename.model
METADATA_FILENAME = CONFIG.filename.metadata
FEATURE_SELECTION_RESULTS_FILENAME = CONFIG.filename.feature_selection_results
SIMULATION_REPORT_FILENAME = CONFIG.filename.simulation_report

TRIPS_SUFFIX = CONFIG.suffix.trips
SUMOCFG_SUFFIX = CONFIG.suffix.sumocfg
FCD_CSV_SUFFIX = CONFIG.suffix.fcd_csv
FCD_PARQUET_SUFFIX = CONFIG.suffix.fcd_parquet

BBOX = CONFIG.network.bbox
ROAD_TYPES = CONFIG.network.road_types
VEHICLE_CLASSES = CONFIG.network.vehicle_classes
NETCONVERT_OPTIONS = CONFIG.network.netconvert_options
POLYCONVERT_OPTIONS = CONFIG.network.polyconvert_options
FRICTION = CONFIG.network.friction
VIEW_SETTINGS = CONFIG.network.view_settings

RANDOM_SEED_DEFAULT = CONFIG.seed.default
RANDOM_SEED_TRAIN = CONFIG.seed.train
RANDOM_SEED_TEST = CONFIG.seed.test
RANDOM_SEED_RAIN = CONFIG.seed.rain

TRAFFIC_GENERATION_PERIODS_MEAN = CONFIG.simulation.traffic_generation_periods_mean
TRAFFIC_GENERATION_PERIODS_STD = CONFIG.simulation.traffic_generation_periods_std
TRAFFIC_GENERATION_PERIODS = CONFIG.simulation.traffic_generation_periods
TLS_ACTUATED_JAM_THRESHOLD = CONFIG.simulation.tls_actuated_jam_threshold
DEVICE_REROUTING_ADAPTATION_STEPS = CONFIG.simulation.device_rerouting_adaptation_steps
DEVICE_REROUTING_ADAPTATION_INTERVAL = CONFIG.simulation.device_rerouting_adaptation_interval
FCD_OUTPUT_ATTRIBUTES = CONFIG.simulation.fcd_output_attributes
START_TIME = CONFIG.simulation.start_time
END_TIME = CONFIG.simulation.end_time
DEVICE_FRICTION_PROBABILITY = CONFIG.simulation.device_friction_probability

ZENODO_DATASET_API_URL = CONFIG.external.zenodo_dataset_api_url

TARGET_COLUMN_ETA = CONFIG.eta.target_column
CHUNK_SIZE = CONFIG.eta.chunk_size
MIN_DURATION = CONFIG.eta.min_duration
MIN_DISTANCE = CONFIG.eta.min_distance
AUGMENTATION_RATE = CONFIG.eta.augmentation_rate
MIN_TRIP_RATIO = CONFIG.eta.min_trip_ratio
NUM_RETRAINING_TRIPS = CONFIG.eta.num_retraining_trips
N_BINS = CONFIG.eta.n_bins
N_SPLITS = CONFIG.eta.n_splits
N_TRAINING_SAMPLES_ETA = CONFIG.eta.n_training_samples
SHRINK_FACTOR_ETA = CONFIG.eta.shrink_factor

MORNING_CEILING = CONFIG.features.morning_ceiling
NOON_FLOOR = CONFIG.features.noon_floor
NOON_CEILING = CONFIG.features.noon_ceiling
AFTERNOON_FLOOR = CONFIG.features.afternoon_floor
RUSH_HOURS = CONFIG.features.rush_hours
PERCENTILE_THRESHOLDS = CONFIG.features.percentile_thresholds
NUM_FREQS = CONFIG.features.num_freqs
COORDINATE_SCALE = CONFIG.features.coordinate_scale
CELL = CONFIG.features.cell
N_CLUSTERS = CONFIG.features.n_clusters
N_COMPONENTS = CONFIG.features.n_components
CORRELATION_THRESHOLD = CONFIG.features.correlation_threshold
PERMUTATION_IMPORTANCE_N_REPEATS = CONFIG.features.permutation_importance_n_repeats
PERMUTATION_IMPORTANCE_SCORING = CONFIG.features.permutation_importance_scoring
PERMUTATION_IMPORTANCE_N_JOBS = CONFIG.features.permutation_importance_n_jobs
RANKING_ALPHA = CONFIG.features.ranking_alpha

OBJECTIVE_XGBOOST = CONFIG.models.objective_xgboost
OBJECTIVE_LIGHTGBM = CONFIG.models.objective_lightgbm
LOSS_FUNCTION_CATBOOST = CONFIG.models.loss_function_catboost
VERBOSE_LIGHTGBM = CONFIG.models.verbose_lightgbm
VERBOSE_CATBOOST = CONFIG.models.verbose_catboost
ENABLE_CATEGORICAL = CONFIG.models.enable_categorical
ALLOW_WRITING_FILES = CONFIG.models.allow_writing_files
TREE_METHOD = CONFIG.models.tree_method
N_ESTIMATORS = CONFIG.models.n_estimators
MAX_DEPTH = CONFIG.models.max_depth
LEARNING_RATE = CONFIG.models.learning_rate
SUBSAMPLE = CONFIG.models.subsample
COLSAMPLE_BYTREE = CONFIG.models.colsample_bytree
N_TRIALS = CONFIG.models.n_trials
N_FOCUSED_TRIALS = CONFIG.models.n_focused_trials
DIRECTION = CONFIG.models.direction
IMPORTANCE_TYPE = CONFIG.models.importance_type

HOST = CONFIG.services.host
ENVIRONMENT = CONFIG.services.environment
LOOP = CONFIG.services.loop
HTTP = CONFIG.services.http
ACCESS_LOG = CONFIG.services.access_log
PORT_BACKEND = CONFIG.services.backend
PORT_PREDICTOR_ETA = CONFIG.services.predictor_eta
PORT_PREDICTOR_FUEL = CONFIG.services.predictor_fuel
PORT_PREDICTOR_STOPS = CONFIG.services.predictor_stops
PORT_DRIFT = CONFIG.services.drift
PORT_FRONTEND = CONFIG.services.frontend
PORT_SUMMARIZER = CONFIG.services.summarizer

SPEED_MULTIPLIER = CONFIG.timelapse.speed_multiplier
INTERVAL_SECONDS = CONFIG.timelapse.interval_seconds
PAUSE_POLL_SECONDS = CONFIG.timelapse.pause_poll_seconds
HTTP_CLIENT_TIMEOUT_SECONDS = CONFIG.timelapse.client_timeout_seconds
COLLECT_SECONDS = CONFIG.timelapse.collect_seconds
METRICS_MAXLEN = CONFIG.timelapse.metrics_maxlen
NOTIFICATIONS_MAXLEN = CONFIG.timelapse.notifications_maxlen

DEFAULT_VERSION = CONFIG.predictor.default_version

SOURCE_X_COLUMN_ETA = CONFIG.predictor_eta.source_x_column
SOURCE_Y_COLUMN_ETA = CONFIG.predictor_eta.source_y_column
DESTINATION_X_COLUMN_ETA = CONFIG.predictor_eta.destination_x_column
DESTINATION_Y_COLUMN_ETA = CONFIG.predictor_eta.destination_y_column
TIME_START_COLUMN_ETA = CONFIG.predictor_eta.time_start_column
DISTANCE_COLUMN_ETA = CONFIG.predictor_eta.distance_column

CONSENSUS_THRESHOLD = CONFIG.drift.consensus_threshold
SMOOTHING_WINDOW_SAMPLES = CONFIG.drift.smoothing_window_samples
GRACE_PERIOD_SAMPLES = CONFIG.drift.grace_period_samples
CALIBRATION_WINDOW_SAMPLES = CONFIG.drift.calibration_window_samples
ADWIN_DELTA_CANDIDATES = CONFIG.drift.adwin_delta_candidates
PAGE_HINKLEY_DELTA_CANDIDATES = CONFIG.drift.page_hinkley_delta_candidates
PAGE_HINKLEY_THRESHOLD_CANDIDATES = CONFIG.drift.page_hinkley_threshold_candidates
KSWIN_ALPHA_CANDIDATES = CONFIG.drift.kswin_alpha_candidates
KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS = CONFIG.drift.kswin_window_size_stat_size_configs
SPC_N_STD_CANDIDATES = CONFIG.drift.spc_n_std_candidates
SPC_MIN_STD = CONFIG.drift.spc_min_std
SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS = (
    CONFIG.drift.spc_consecutive_violations_required_multiplier_configs
)

FEATURE_CATEGORIES = CONFIG.feature_categories

TIME_START_COLUMN_FUEL = CONFIG.fuel.time_start_column
TARGET_COLUMN_FUEL = CONFIG.fuel.target_column
RANDOM_SEED_FUEL = CONFIG.fuel.random_seed
MIN_TRIP_POINTS = CONFIG.fuel.min_trip_points
N_START_CLUSTERS = CONFIG.fuel.n_start_clusters
N_END_CLUSTERS = CONFIG.fuel.n_end_clusters
N_INIT = CONFIG.fuel.n_init
START_HOUR_MAX = CONFIG.fuel.start_hour_max

TIME_START_COLUMN_STOPS = CONFIG.stops.time_start_column
TARGET_COLUMN_STOPS = CONFIG.stops.target_column
RANDOM_SEED_STOPS = CONFIG.stops.random_seed
N_CLUSTERS_STOPS = CONFIG.stops.n_clusters
N_INIT_STOPS = CONFIG.stops.n_init
MIN_TRIP_RECORDS_STOPS = CONFIG.stops.min_trip_records
MIN_DURATION_STOPS = CONFIG.stops.min_duration
MIN_DISTANCE_STOPS = CONFIG.stops.min_distance
USE_SPATIAL_FEATURES_STOPS = CONFIG.stops.use_spatial_features
USE_ROUTE_FEATURES_STOPS = CONFIG.stops.use_route_features

MAX_RETRIES = CONFIG.summarizer.max_retries
RETRY_DELAY_SECONDS = CONFIG.summarizer.retry_delay_seconds
ASYNC_CLIENT_TIMEOUT_SECONDS = CONFIG.summarizer.async_client_timeout_seconds
OPENROUTER_BASE_URL = CONFIG.summarizer.openrouter_base_url
OPENROUTER_MODEL = CONFIG.summarizer.openrouter_model
SUMMARIZER_SYSTEM_PROMPT = CONFIG.summarizer.system_prompt
