"""
Configuration management module for the project.
Provides centralized configuration loading, dataclass definitions for all config sections, and global constants used throughout the project.
"""

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
XML2CSV = SUMO_TOOLS / "xml" / "xml2csv.py"
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


@dataclass(frozen=True, slots=True)
class SuffixConfig:
    trips: str
    fcd_xml: str
    sumocfg: str
    fcd_csv: str


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
    traffic_generation_periods_noise: float
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
    min_duration: int
    min_distance: int
    augmentation_rate: float
    min_trip_ratio: float
    num_retraining_trips: int


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
        )


CONFIG_PATH = Path(__file__).parent / "config.yaml"
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

OSM_DATA_FILENAME = CONFIG.filename.osm_data
GUI_SETTINGS_FILENAME = CONFIG.filename.gui_settings
POLY_FILENAME = CONFIG.filename.poly
NETWORK_BASE_FILENAME = CONFIG.filename.network_base
NETWORK_RAIN_FILENAME = CONFIG.filename.network_rain
ROUTES_TEMP_FILENAME = CONFIG.filename.routes_temp
LOGS_FILENAME = CONFIG.filename.logs
RESULTS_FILENAME = CONFIG.filename.results
RESEARCH_RESULTS_FILENAME = CONFIG.filename.research_results

TRIPS_SUFFIX = CONFIG.suffix.trips
FCD_XML_SUFFIX = CONFIG.suffix.fcd_xml
SUMOCFG_SUFFIX = CONFIG.suffix.sumocfg
FCD_CSV_SUFFIX = CONFIG.suffix.fcd_csv

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

TRAFFIC_GENERATION_PERIODS_NOISE = CONFIG.simulation.traffic_generation_periods_noise
TRAFFIC_GENERATION_PERIODS = CONFIG.simulation.traffic_generation_periods
TLS_ACTUATED_JAM_THRESHOLD = CONFIG.simulation.tls_actuated_jam_threshold
DEVICE_REROUTING_ADAPTATION_STEPS = CONFIG.simulation.device_rerouting_adaptation_steps
DEVICE_REROUTING_ADAPTATION_INTERVAL = CONFIG.simulation.device_rerouting_adaptation_interval
FCD_OUTPUT_ATTRIBUTES = CONFIG.simulation.fcd_output_attributes
START_TIME = CONFIG.simulation.start_time
END_TIME = CONFIG.simulation.end_time
DEVICE_FRICTION_PROBABILITY = CONFIG.simulation.device_friction_probability

ZENODO_DATASET_API_URL = CONFIG.external.zenodo_dataset_api_url

MIN_DURATION = CONFIG.eta.min_duration
MIN_DISTANCE = CONFIG.eta.min_distance
AUGMENTATION_RATE = CONFIG.eta.augmentation_rate
MIN_TRIP_RATIO = CONFIG.eta.min_trip_ratio
NUM_RETRAINING_TRIPS = CONFIG.eta.num_retraining_trips
