import os
from pathlib import Path

import numpy as np

np.random.seed(42)
TRAIN_SEED = 42
TEST_SEED = 123

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.45, 0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.55, 0.50, 0.55]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.uniform(0.98, 1.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]

SUMO_HOME_ENV = os.environ.get("SUMO_HOME", None)
if SUMO_HOME_ENV is None:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set. Please run `conda deactivate && conda activate thesis` to set up the SUMO_HOME environment variable, and to add SUMO_HOME/bin and SUMO_HOME/tools to your PATH."
    )
SUMO_HOME = Path(SUMO_HOME_ENV)

OSM_WEB_WIZARD = SUMO_HOME / "tools" / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
DUAROUTER = SUMO_HOME / "bin" / ("duarouter.exe" if os.name == "nt" else "duarouter")
XML2CSV = SUMO_HOME / "tools" / "xml" / "xml2csv.py"

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_LOGGER_NAME = "dataset"
LOGGER_NAMES = [
    DATASET_LOGGER_NAME,
]
LOG_FILES_CONFIG = {name: LOGS_DIR / f"{name}.log" for name in LOGGER_NAMES}

DATASET_DIR = PROJECT_ROOT / "src" / "dataset"
SIMULATION_DIR = DATASET_DIR / "athens-10h"
PLOTS_DIR = SIMULATION_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NETWORK = SIMULATION_DIR / "osm.net.xml.gz"
FIXED_FLOWS_FILE = SIMULATION_DIR / "fixed.flows.xml"
FIXED_ROUTES_FILE = SIMULATION_DIR / "fixed.rou.xml"
FIXED_ROUTES_ALT_FILE = SIMULATION_DIR / "fixed.rou.alt.xml"

VEHICLE_TYPE_CAR = "car"
VEHICLE_TYPE_CAR_RAIN = "car-rain"

SCENARIOS = {
    "base": {"vehicle_type": VEHICLE_TYPE_CAR},
    "closure": {"vehicle_type": VEHICLE_TYPE_CAR},
    "rain": {"vehicle_type": VEHICLE_TYPE_CAR_RAIN},
}

DEFAULT_FLAGS = {
    "gui": False,
    "convert": False,
    "delete_original": False,
}

DATASET_SPECS = [
    {
        "name": name,
        "train": {
            "dataset_id": f"{name}-train",
            "network": NETWORK,
            "trips_file": SIMULATION_DIR / f"{name}-train.trips.xml",
            "traffic_generation_periods": TRAIN_TRAFFIC_GENERATION_PERIODS,
            "seed": TRAIN_SEED,
            "config": SIMULATION_DIR / f"{name}-train.sumocfg",
            "fcd_output": SIMULATION_DIR / f"{name}-train-fcd.xml",
            "fixed_routes_file": None,
            **extra_specs,
            **DEFAULT_FLAGS,
        },
        "test": {
            "dataset_id": f"{name}-test",
            "network": NETWORK,
            "trips_file": SIMULATION_DIR / f"{name}-test.trips.xml",
            "traffic_generation_periods": TEST_TRAFFIC_GENERATION_PERIODS,
            "seed": TEST_SEED,
            "config": SIMULATION_DIR / f"{name}-test.sumocfg",
            "fcd_output": SIMULATION_DIR / f"{name}-test-fcd.xml",
            "fixed_routes_file": FIXED_ROUTES_FILE,
            **extra_specs,
            **DEFAULT_FLAGS,
        },
    }
    for name, extra_specs in SCENARIOS.items()
]
