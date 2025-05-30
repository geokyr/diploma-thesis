import os
from pathlib import Path

import numpy as np

np.random.seed(42)
TRAIN_SEED = 42
TEST_SEED = 123

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.45, 0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.55, 0.50, 0.55]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.uniform(0.98, 1.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]
# TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.normal(1, 0.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]

SUMO_HOME = Path(os.environ.get("SUMO_HOME", None))
if SUMO_HOME is None:
    raise EnvironmentError(
        "Please set the SUMO_HOME environment variable. Also add SUMO_HOME/bin and SUMO_HOME/tools to your PATH."
    )

OSM_WEB_WIZARD = SUMO_HOME / "tools" / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
DUAROUTER = SUMO_HOME / "bin" / ("duarouter.exe" if os.name == "nt" else "duarouter")
XML2CSV = SUMO_HOME / "tools" / "xml" / "xml2csv.py"

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / "athens-10h"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "dataset.log"
NETWORK = DATA_DIR / "osm.net.xml.gz"
FIXED_FLOWS_FILE = DATA_DIR / "fixed.flows.xml"
FIXED_ROUTES_FILE = DATA_DIR / "fixed.rou.xml"
FIXED_ROUTES_ALT_FILE = DATA_DIR / "fixed.rou.alt.xml"

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
            "trips_file": DATA_DIR / f"{name}-train.trips.xml",
            "traffic_generation_periods": TRAIN_TRAFFIC_GENERATION_PERIODS,
            "seed": TRAIN_SEED,
            "config": DATA_DIR / f"{name}-train.sumocfg",
            "fcd_output": DATA_DIR / f"{name}-train-fcd.xml",
            "fixed_routes_file": None,
            **extra_specs,
            **DEFAULT_FLAGS,
        },
        "test": {
            "dataset_id": f"{name}-test",
            "network": NETWORK,
            "trips_file": DATA_DIR / f"{name}-test.trips.xml",
            "traffic_generation_periods": TEST_TRAFFIC_GENERATION_PERIODS,
            "seed": TEST_SEED,
            "config": DATA_DIR / f"{name}-test.sumocfg",
            "fcd_output": DATA_DIR / f"{name}-test-fcd.xml",
            "fixed_routes_file": FIXED_ROUTES_FILE,
            **extra_specs,
            **DEFAULT_FLAGS,
        },
    }
    for name, extra_specs in SCENARIOS.items()
]
