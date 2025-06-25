import os
from pathlib import Path

from thesis.common.config import (
    DATA_DIR,
    SIMULATION_DIR,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
    TYPE_TEST,
    TYPE_TRAIN,
)

TRAIN_SEED = 42
TEST_SEED = 123

SUMO_HOME_ENV = os.environ.get("SUMO_HOME", None)
if SUMO_HOME_ENV is None:
    raise EnvironmentError(
        "SUMO_HOME environment variable is not set, please run `conda deactivate && conda activate thesis`"
    )
SUMO_HOME = Path(SUMO_HOME_ENV)

OSM_WEB_WIZARD = SUMO_HOME / "tools" / "osmWebWizard.py"
RANDOM_TRIPS = SUMO_HOME / "tools" / "randomTrips.py"
DUAROUTER = SUMO_HOME / "bin" / ("duarouter.exe" if os.name == "nt" else "duarouter")
XML2CSV = SUMO_HOME / "tools" / "xml" / "xml2csv.py"

LOGS_DIR = SIMULATION_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NETWORK = SIMULATION_DIR / "osm.net.xml.gz"
FIXED_FLOWS_FILE = SIMULATION_DIR / "fixed.flows.xml"
FIXED_ROUTES_FILE = SIMULATION_DIR / "fixed.rou.xml"
FIXED_ROUTES_ALT_FILE = SIMULATION_DIR / "fixed.rou.alt.xml"

VEHICLE_TYPE_CAR = "car"
VEHICLE_TYPE_CAR_RAIN = "car-rain"
SCENARIO_CONFIGS = {
    "base": {
        "vehicle_type": VEHICLE_TYPE_CAR,
    },
    "closure": {
        "vehicle_type": VEHICLE_TYPE_CAR,
    },
    "rain": {
        "vehicle_type": VEHICLE_TYPE_CAR_RAIN,
    },
}

DATASET_SPECS = {}
for scenario_name, scenario_config in SCENARIO_CONFIGS.items():
    for type in [TYPE_TRAIN, TYPE_TEST]:
        dataset_name = f"{scenario_name}-{type}"
        DATASET_SPECS[dataset_name] = {
            "trips_file": SIMULATION_DIR / f"{dataset_name}.trips.xml",
            "traffic_generation_periods": (
                TRAIN_TRAFFIC_GENERATION_PERIODS if type == TYPE_TRAIN else TEST_TRAFFIC_GENERATION_PERIODS
            ),
            "seed": TRAIN_SEED if type == TYPE_TRAIN else TEST_SEED,
            "config": SIMULATION_DIR / f"{dataset_name}.sumocfg",
            "fcd_output": DATA_DIR / f"{dataset_name}-fcd.xml",
            "fixed_routes_file": None if type == TYPE_TRAIN else FIXED_ROUTES_FILE,
            **scenario_config,
        }
