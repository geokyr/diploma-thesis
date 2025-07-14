import os
import sys
from pathlib import Path

from thesis.common.config import SIMULATION_DIR, TYPE_TEST, TYPE_TRAIN

SEED_BASE = 42
SEED_RAIN = 1337

DEVICE_FRICTION_PROBABILITY = 1.0

VIRTUAL_ENV = os.environ.get("VIRTUAL_ENV", None)
if VIRTUAL_ENV is None:
    raise EnvironmentError("VIRTUAL_ENV environment variable is not set, please activate the virtual environment")
VIRTUAL_ENV = Path(VIRTUAL_ENV)

PYTHON_VERSION = f"python{sys.version_info.major}.{sys.version_info.minor}"
SUMO_HOME_CANDIDATES = [
    VIRTUAL_ENV / "lib" / PYTHON_VERSION / "site-packages" / "sumo",
    VIRTUAL_ENV / "Lib" / "site-packages" / "sumo",
]
SUMO_HOME = next((C for C in SUMO_HOME_CANDIDATES if C.exists()), None)
if SUMO_HOME is None:
    raise EnvironmentError(f"Could not locate the SUMO installation in your venv. Tried: {SUMO_HOME_CANDIDATES}")
SUMO_HOME = Path(SUMO_HOME)
os.environ["SUMO_HOME"] = str(SUMO_HOME)

SUMO_BIN = SUMO_HOME / "bin"
SUMO_TOOLS = SUMO_HOME / "tools"
PATHS = [os.environ.get("PATH", ""), str(SUMO_BIN), str(SUMO_TOOLS)]
os.environ["PATH"] = os.pathsep.join(PATHS)

OSM_WEB_WIZARD = SUMO_TOOLS / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_TOOLS / "randomTrips.py"
XML2CSV = SUMO_TOOLS / "xml" / "xml2csv.py"

LOGS_DIR = SIMULATION_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_BASE = SIMULATION_DIR / "osm.net.xml.gz"
NETWORK_RAIN = SIMULATION_DIR / "osm-rain.net.xml.gz"

SCENARIO_BASE = "base"
SCENARIO_RAIN = "rain"
OUTPUT_EMISSION = "emission"
OUTPUT_FCD = "fcd"

SCENARIOS = [SCENARIO_BASE, SCENARIO_RAIN]
OUTPUTS = [OUTPUT_EMISSION, OUTPUT_FCD]
TYPES = [TYPE_TRAIN, TYPE_TEST]
