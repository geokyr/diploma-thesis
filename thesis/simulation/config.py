import os
from pathlib import Path

from thesis.common.config import SIMULATION_DIR, TYPE_TEST, TYPE_TRAIN

SEED_BASE = 42
SEED_RAIN = 1337

DEVICE_FRICTION_PROBABILITY = 1.0

SUMO_HOME_ENV = os.environ.get("SUMO_HOME", None)
if SUMO_HOME_ENV is None:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set, please run `conda deactivate && conda activate thesis`"
    )
SUMO_HOME = Path(SUMO_HOME_ENV)

OSM_WEB_WIZARD = SUMO_HOME / "tools" / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
XML2CSV = SUMO_HOME / "tools" / "xml" / "xml2csv.py"

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
