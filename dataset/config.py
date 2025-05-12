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
TRAIN_FCD = DATA_DIR / "train-fcd.xml"
TEST_FCD = DATA_DIR / "test-fcd.xml"
CLOSURE_TRAIN_FCD = DATA_DIR / "closure-train-fcd.xml"
CLOSURE_TEST_FCD = DATA_DIR / "closure-test-fcd.xml"
RAIN_TRAIN_FCD = DATA_DIR / "rain-train-fcd.xml"
RAIN_TEST_FCD = DATA_DIR / "rain-test-fcd.xml"

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.45, 0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.55, 0.50, 0.55]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.uniform(0.98, 1.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]
