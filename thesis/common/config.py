import os
from pathlib import Path
from sysconfig import get_paths

MAX_FILE_SIZE = 10 * 1024 * 1024
BACKUP_COUNT = 5

PROJECT_DIR = Path(__file__).parent.parent.parent
SIMULATION_DIR = PROJECT_DIR / "simulation"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

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

DATA_DIRNAME = "data"
LOGS_DIRNAME = "logs"
PLOTS_DIRNAME = "plots"
OSM_DATA_FILENAME = "osm_bbox.osm.xml.gz"
GUI_SETTINGS_FILENAME = "osm.view.xml"
POLY_FILENAME = "osm.poly.xml.gz"
NETWORK_BASE_FILENAME = "osm.net.xml.gz"
NETWORK_RAIN_FILENAME = "osm-rain.net.xml.gz"

BBOX = (23.725252771719436, 37.974745936977456, 23.752735758169127, 37.988290142332225)
ROAD_TYPES = '{"Highway": ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "unsurfaced", "service", "raceway", "bus_guideway", "track", "footway", "pedestrian", "path", "bridleway", "cycleway", "step", "steps", "stairs"], "Railway": ["preserved", "tram", "subway", "light_rail", "rail", "highspeed", "monorail"], "Aeroway": ["stopway", "parking_position", "taxiway", "taxilane", "runway", "highway_strip"], "Waterway": ["river", "canal"], "Aerialway": ["cable_car", "gondola"], "Route": ["ferry"]}'
VEHICLE_CLASSES = "passenger"
NETCONVERT_TYPEMAP = SUMO_HOME / "data" / "typemap" / "osmNetconvert.typ.xml"
POLYCONVERT_TYPEMAP = SUMO_HOME / "data" / "typemap" / "osmPolyconvert.typ.xml"
NETCONVERT_OPTIONS = "--geometry.remove,--roundabouts.guess,--ramps.guess,--junctions.join,--tls.guess-signals,--tls.discard-simple,--tls.join,--output.original-names,--junctions.corner-detail,5,--output.street-names,--tls.default-type,actuated"
POLYCONVERT_OPTIONS = "--verbose,--osm.keep-full-type,--osm.merge-relations,1"
FRICTION = 0.4
VIEW_SETTINGS = """<viewsettings>
    <scheme name="real world"/>
    <delay value="20"/>
</viewsettings>
"""

TRIPS_SUFFIX = ".trips.xml"
FCD_XML_SUFFIX = "-fcd.xml"
SUMOCFG_SUFFIX = ".sumocfg"
FCD_CSV_SUFFIX = "-fcd.csv"

RANDOM_SEED = 42
RANDOM_SEED_TRAIN = 13
RANDOM_SEED_TEST = 2025
RANDOM_SEED_RAIN = 314159
TRAFFIC_GENERATION_PERIODS_NOISE = 0.01
TRAFFIC_GENERATION_PERIODS = (0.50, 0.55, 0.65, 0.75, 0.80, 0.80, 0.75, 0.65, 0.65, 0.60)

TLS_ACTUATED_JAM_THRESHOLD = 30
DEVICE_REROUTING_ADAPTATION_STEPS = 18
DEVICE_REROUTING_ADAPTATION_INTERVAL = 10
FCD_OUTPUT_ATTRIBUTES = "id,x,y,speed,lane,odometer,fuel,waiting"
START_TIME = 0
END_TIME = 36000
ROUTES_TEMP_FILENAME = "routes.rou.xml"
DEVICE_FRICTION_PROBABILITY = 1.0

ZENODO_DATASET_API_URL = "https://zenodo.org/api/records/15916712"

MODELS_DIRNAME = "models"
RESULTS_DIRNAME = "results"

MIN_DURATION = 30
MIN_DISTANCE = 200
AUGMENTATION_RATE = 0.0
MIN_TRIP_RATIO = 0.5

NUM_RETRAINING_TRIPS = 10000
