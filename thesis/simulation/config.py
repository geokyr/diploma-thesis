import os
from pathlib import Path

from thesis.common.config import SIMULATION_DIR, TYPE_TEST, TYPE_TRAIN

SEED_BASE = 42
SEED_CLOSURE = 123
SEED_RAIN = 1337

DEVICE_FRICTION_PROBABILITY = 1.0

CLOSURE_EDGES = [
    "10741754#0",
    "168914458#0",
    "168914459#0",
    "222899032#0",
    "222899032#1",
    "23182806#0",
    "260124786#0",
    "260124786#2",
    "284503674#0",
    "298828555#0",
    "299510529#0",
    "299644700#0",
    "299644700#2",
    "299644700#5",
    "299644700#7",
    "299644700#8",
    "32245113#0",
    "32295798#0",
    "52493436#0",
    "52493436#12",
    "52493436#5",
    "52493436#8",
    "52493441#0",
    "751068056",
    "972965170#0",
]

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
NETWORK_CLOSURE = SIMULATION_DIR / "osm-closure.net.xml.gz"

SCENARIO_BASE = "base"
SCENARIO_CLOSURE = "closure"
SCENARIO_RAIN = "rain"
OUTPUT_EMISSION = "emission"
OUTPUT_FCD = "fcd"

SCENARIOS = [SCENARIO_BASE, SCENARIO_CLOSURE, SCENARIO_RAIN]
OUTPUTS = [OUTPUT_EMISSION, OUTPUT_FCD]
TYPES = [TYPE_TRAIN, TYPE_TEST]
