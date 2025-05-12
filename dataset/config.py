import os
from pathlib import Path

import numpy as np

np.random.seed(42)
TRAIN_SEED = 42
TEST_SEED = 123

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
DATA_DIR = BASE_DIR / "athens-10h"
PLOTS_DIR = DATA_DIR / "plots"

NETWORK = DATA_DIR / "osm.net.xml.gz"
FIXED_FLOWS_FILE = DATA_DIR / "fixed.flows.xml"
FIXED_ROUTES_FILE = DATA_DIR / "fixed.rou.xml"
FIXED_ROUTES_ALT_FILE = DATA_DIR / "fixed.rou.alt.xml"

BASE_TRAIN_TRIPS_FILE = DATA_DIR / "base-train.trips.xml"
BASE_TEST_TRIPS_FILE = DATA_DIR / "base-test.trips.xml"
BASE_TRAIN_SIMULATION_CONFIG = DATA_DIR / "base-train.sumocfg"
BASE_TEST_SIMULATION_CONFIG = DATA_DIR / "base-test.sumocfg"
BASE_TRAIN_FCD = DATA_DIR / "base-train-fcd.xml"
BASE_TEST_FCD = DATA_DIR / "base-test-fcd.xml"
BASE_TRAIN_DATASET_ID = "base-train"
BASE_TEST_DATASET_ID = "base-test"

CLOSURE_TRAIN_TRIPS_FILE = DATA_DIR / "closure-train.trips.xml"
CLOSURE_TEST_TRIPS_FILE = DATA_DIR / "closure-test.trips.xml"
CLOSURE_TRAIN_SIMULATION_CONFIG = DATA_DIR / "closure-train.sumocfg"
CLOSURE_TEST_SIMULATION_CONFIG = DATA_DIR / "closure-test.sumocfg"
CLOSURE_TRAIN_FCD = DATA_DIR / "closure-train-fcd.xml"
CLOSURE_TEST_FCD = DATA_DIR / "closure-test-fcd.xml"
CLOSURE_TRAIN_DATASET_ID = "closure-train"
CLOSURE_TEST_DATASET_ID = "closure-test"

RAIN_TRAIN_TRIPS_FILE = DATA_DIR / "rain-train.trips.xml"
RAIN_TEST_TRIPS_FILE = DATA_DIR / "rain-test.trips.xml"
RAIN_TRAIN_SIMULATION_CONFIG = DATA_DIR / "rain-train.sumocfg"
RAIN_TEST_SIMULATION_CONFIG = DATA_DIR / "rain-test.sumocfg"
RAIN_TRAIN_FCD = DATA_DIR / "rain-train-fcd.xml"
RAIN_TEST_FCD = DATA_DIR / "rain-test-fcd.xml"
RAIN_TRAIN_DATASET_ID = "rain-train"
RAIN_TEST_DATASET_ID = "rain-test"

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.45, 0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.55, 0.50, 0.55]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.uniform(0.98, 1.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]
