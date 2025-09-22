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
    latest: str
    model: str
    metadata: str


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
    chunk_size: int
    min_duration: int
    min_distance: int
    augmentation_rate: float
    min_trip_ratio: float
    num_retraining_trips: int
    n_bins: int
    n_splits: int


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    morning_ceiling: int
    noon_floor: int
    noon_ceiling: int
    afternoon_floor: int
    rush_hours: list[int]
    percentile_thresholds: list[int]
    num_freqs: int
    coordinate_scale: float
    cell: int
    n_clusters: int
    n_components: int


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    objective_xgboost: str
    objective_lightgbm: str
    loss_function_catboost: str
    verbose_lightgbm: int
    verbose_catboost: int
    enable_categorical: bool
    allow_writing_files: bool
    max_cat_to_onehot: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    n_trials: int
    direction: str


@dataclass(frozen=True, slots=True)
class ServicesConfig:
    host: str
    environment: str
    backend: int
    predictor_eta: int
    predictor_fuel: int
    predictor_stops: int
    drift: int
    frontend: int


@dataclass(frozen=True, slots=True)
class TimelapseConfig:
    speed_multiplier: float
    interval_ms: int
    max_intervals: int
    client_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class PredictorConfig:
    time_start_column: str
    latest_version: str


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
LATEST_FILENAME = CONFIG.filename.latest
MODEL_FILENAME = CONFIG.filename.model
METADATA_FILENAME = CONFIG.filename.metadata

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

CHUNK_SIZE = CONFIG.eta.chunk_size
MIN_DURATION = CONFIG.eta.min_duration
MIN_DISTANCE = CONFIG.eta.min_distance
AUGMENTATION_RATE = CONFIG.eta.augmentation_rate
MIN_TRIP_RATIO = CONFIG.eta.min_trip_ratio
NUM_RETRAINING_TRIPS = CONFIG.eta.num_retraining_trips
N_BINS = CONFIG.eta.n_bins
N_SPLITS = CONFIG.eta.n_splits

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

OBJECTIVE_XGBOOST = CONFIG.models.objective_xgboost
OBJECTIVE_LIGHTGBM = CONFIG.models.objective_lightgbm
LOSS_FUNCTION_CATBOOST = CONFIG.models.loss_function_catboost
VERBOSE_LIGHTGBM = CONFIG.models.verbose_lightgbm
VERBOSE_CATBOOST = CONFIG.models.verbose_catboost
ENABLE_CATEGORICAL = CONFIG.models.enable_categorical
ALLOW_WRITING_FILES = CONFIG.models.allow_writing_files
MAX_CAT_TO_ONEHOT = CONFIG.models.max_cat_to_onehot
N_ESTIMATORS = CONFIG.models.n_estimators
MAX_DEPTH = CONFIG.models.max_depth
LEARNING_RATE = CONFIG.models.learning_rate
SUBSAMPLE = CONFIG.models.subsample
COLSAMPLE_BYTREE = CONFIG.models.colsample_bytree
N_TRIALS = CONFIG.models.n_trials
DIRECTION = CONFIG.models.direction

HOST = CONFIG.services.host
ENVIRONMENT = CONFIG.services.environment
PORT_BACKEND = CONFIG.services.backend
PORT_PREDICTOR_ETA = CONFIG.services.predictor_eta
PORT_PREDICTOR_FUEL = CONFIG.services.predictor_fuel
PORT_PREDICTOR_STOPS = CONFIG.services.predictor_stops
PORT_DRIFT = CONFIG.services.drift
PORT_FRONTEND = CONFIG.services.frontend

SPEED_MULTIPLIER = CONFIG.timelapse.speed_multiplier
INTERVAL_MS = CONFIG.timelapse.interval_ms
MAX_INTERVALS = CONFIG.timelapse.max_intervals
HTTP_CLIENT_TIMEOUT_SECONDS = CONFIG.timelapse.client_timeout_seconds

TIME_START_COLUMN = CONFIG.predictor.time_start_column
LATEST_VERSION = CONFIG.predictor.latest_version
