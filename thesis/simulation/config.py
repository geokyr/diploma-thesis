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
        "SUMO_HOME environment variable is not set. Please run `conda deactivate && conda activate thesis`. This will set up the SUMO_HOME environment variable, and add SUMO_HOME/bin and SUMO_HOME/tools to your PATH."
    )
SUMO_HOME = Path(SUMO_HOME_ENV)

OSM_WEB_WIZARD = SUMO_HOME / "tools" / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
DUAROUTER = SUMO_HOME / "bin" / ("duarouter.exe" if os.name == "nt" else "duarouter")
XML2CSV = SUMO_HOME / "tools" / "xml" / "xml2csv.py"

PROJECT_ROOT = Path(__file__).parent.parent.parent
SIMULATION_DIR = PROJECT_ROOT / "simulation"
LOGS_DIR = SIMULATION_DIR / "logs"
PLOTS_DIR = SIMULATION_DIR / "plots"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NETWORK = SIMULATION_DIR / "osm.net.xml.gz"
FIXED_FLOWS_FILE = SIMULATION_DIR / "fixed.flows.xml"
FIXED_ROUTES_FILE = SIMULATION_DIR / "fixed.rou.xml"
FIXED_ROUTES_ALT_FILE = SIMULATION_DIR / "fixed.rou.alt.xml"

DATASET_TYPE_TRAIN = "train"
DATASET_TYPE_TEST = "test"
DATASET_TYPES = [DATASET_TYPE_TRAIN, DATASET_TYPE_TEST]
VEHICLE_TYPE_CAR = "car"
VEHICLE_TYPE_CAR_RAIN = "car-rain"
DEFAULT_CONFIG = {
    "gui": False,
    "convert": False,
    "delete_original": False,
}
SCENARIO_CONFIGS = {
    "base": {
        "vehicle_type": VEHICLE_TYPE_CAR,
        **DEFAULT_CONFIG,
    },
    "closure": {
        "vehicle_type": VEHICLE_TYPE_CAR,
        **DEFAULT_CONFIG,
    },
    "rain": {
        "vehicle_type": VEHICLE_TYPE_CAR_RAIN,
        **DEFAULT_CONFIG,
    },
}

DATASET_SPECS = {}
for scenario_name, scenario_config in SCENARIO_CONFIGS.items():
    for dataset_type in DATASET_TYPES:
        dataset_key = f"{scenario_name}-{dataset_type}"
        DATASET_SPECS[dataset_key] = {
            "dataset_id": dataset_key,
            "trips_file": SIMULATION_DIR / f"{dataset_key}.trips.xml",
            "traffic_generation_periods": TRAIN_TRAFFIC_GENERATION_PERIODS
            if dataset_type == DATASET_TYPE_TRAIN
            else TEST_TRAFFIC_GENERATION_PERIODS,
            "seed": TRAIN_SEED if dataset_type == DATASET_TYPE_TRAIN else TEST_SEED,
            "config": SIMULATION_DIR / f"{dataset_key}.sumocfg",
            "fcd_output": SIMULATION_DIR / f"{dataset_key}-fcd.xml",
            "fixed_routes_file": None if dataset_type == DATASET_TYPE_TRAIN else FIXED_ROUTES_FILE,
            **scenario_config,
        }
