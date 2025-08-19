"""
SUMO simulation pipeline orchestration and command execution utilities.
Provides functions for downloading OSM data, building networks, generating trips, running simulations, and converting output data formats.
"""

import gzip
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from thesis.common.config import (
    BBOX,
    DEVICE_FRICTION_PROBABILITY,
    DEVICE_REROUTING_ADAPTATION_INTERVAL,
    DEVICE_REROUTING_ADAPTATION_STEPS,
    END_TIME,
    FCD_OUTPUT_ATTRIBUTES,
    FRICTION,
    NETCONVERT_OPTIONS,
    NETCONVERT_TYPEMAP,
    OSM_BUILD,
    OSM_GET,
    POLYCONVERT_OPTIONS,
    POLYCONVERT_TYPEMAP,
    RANDOM_TRIPS,
    ROAD_TYPES,
    ROUTES_TEMP_FILENAME,
    START_TIME,
    TLS_ACTUATED_JAM_THRESHOLD,
    VEHICLE_CLASSES,
    VIEW_SETTINGS,
    XML2CSV,
)

logger = logging.getLogger(__name__)


def _validate_path(path: Path, description: str) -> None:
    """
    Validate a path and raise an error if not found.

    Args:
        path (Path): Path to validate.
        description (str): Description of the path.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not path.exists():
        error_msg = f"{description} not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def _execute_command(command: list[str], name: str) -> None:
    """
    Execute a command.

    Args:
        command (list[str]): Command to execute.
        name (str): Name of the command.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
    """
    command_str = " ".join(str(arg) for arg in command)
    logger.info(f"[{name}] Executing: {command_str}")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
        )

        for line in process.stdout:
            if line.rstrip():
                logger.info(f"[{name}] {line.rstrip()}")

        return_code = process.wait()
        if return_code != 0:
            logger.error(f"[{name}] Failed with return code {return_code}")
            raise subprocess.CalledProcessError(return_code, command)

        logger.info(f"[{name}] Completed successfully")

    except subprocess.CalledProcessError:
        raise
    except Exception as e:
        logger.exception(f"[{name}] Unexpected error during execution: {e}")
        raise


def get_osm_data(
    simulation_dir: Path,
    bbox: list[float] = BBOX,
    road_types: str = ROAD_TYPES,
) -> None:
    """
    Get the osm data using the sumo osmGet tool.

    Args:
        simulation_dir (Path): Directory to save the osm data.
        bbox (tuple[float, float, float, float]): Bounding box of the area to get osm data from.
        road_types (str): Road types to get osm data from.

    Raises:
        FileNotFoundError: If the simulation directory does not exist.
    """
    _validate_path(simulation_dir, "Simulation directory")

    bbox_str = ",".join(str(v) for v in bbox)
    command = [
        "python",
        str(OSM_GET),
        "--output-dir",
        str(simulation_dir),
        "--bbox",
        bbox_str,
        "--road-types",
        road_types,
        "--shapes",
        "--gzip",
    ]
    name = "osmGet"
    _execute_command(command, name)


def build_network(
    simulation_dir: Path,
    osm_data_path: Path,
    vehicle_classes: str = VEHICLE_CLASSES,
    netconvert_typemap: Path = NETCONVERT_TYPEMAP,
    polyconvert_typemap: Path = POLYCONVERT_TYPEMAP,
    netconvert_options: str = NETCONVERT_OPTIONS,
    polyconvert_options: str = POLYCONVERT_OPTIONS,
) -> None:
    """
    Build the network using the sumo osmBuild tool.

    Args:
        simulation_dir (Path): Directory to save the network.
        osm_data_path (Path): Path to the osm data file.
        vehicle_classes (str): Vehicle classes to include in the network.
        netconvert_typemap (Path): Path to the netconvert typemap file.
        polyconvert_typemap (Path): Path to the polyconvert typemap file.
        netconvert_options (str): Options for the netconvert tool.
        polyconvert_options (str): Options for the polyconvert tool.

    Raises:
        FileNotFoundError: If the simulation directory, osm data file, netconvert typemap file, or polyconvert typemap file does not exist.
    """
    _validate_path(simulation_dir, "Simulation directory")
    _validate_path(osm_data_path, "OSM data file")
    _validate_path(netconvert_typemap, "Netconvert typemap file")
    _validate_path(polyconvert_typemap, "Polyconvert typemap file")

    command = [
        "python",
        str(OSM_BUILD),
        "--osm-file",
        str(osm_data_path),
        "--output-directory",
        str(simulation_dir),
        "--vehicle-classes",
        vehicle_classes,
        "--netconvert-typemap",
        str(netconvert_typemap),
        "--typemap",
        str(polyconvert_typemap),
        f"--netconvert-options={netconvert_options}",
        f"--polyconvert-options={polyconvert_options}",
        "--gzip",
    ]
    name = "osmBuild"
    _execute_command(command, name)


def build_rain_network(network_base_path: Path, network_rain_path: Path, friction: float = FRICTION) -> None:
    """
    Build a rain network with reduced friction.

    Args:
        network_base_path (Path): Path to the network file.
        network_rain_path (Path): Path to the rain network file.
        friction (float): Friction value to apply to all edges.

    Raises:
        FileNotFoundError: If the base network file does not exist.
    """
    _validate_path(network_base_path, "Base network file")

    logger.info(f"Building rain network with friction {friction}")

    with gzip.open(network_base_path, "rb") as f_in:
        tree = ET.parse(f_in)
        root = tree.getroot()

    for lane in root.findall(".//lane"):
        lane.set("friction", str(friction))

    with gzip.open(network_rain_path, "wb") as f_out:
        tree.write(f_out, xml_declaration=True, encoding="UTF-8")

    logger.info("Built rain network successfully")


def write_gui_settings_file(gui_settings_path: Path) -> None:
    """
    Write the GUI settings file.

    Args:
        gui_settings_path (Path): Path to the GUI settings file.
    """
    logger.info("Writing GUI settings file")

    gui_settings_path.write_text(VIEW_SETTINGS, encoding="utf-8")

    logger.info("Wrote GUI settings file successfully")


def create_configuration_file(
    network_path: Path,
    trips_path: Path,
    poly_path: Path,
    gui_settings_path: Path,
    fcd_xml_path: Path,
    sumocfg_path: Path,
    tls_actuated_jam_threshold: int = TLS_ACTUATED_JAM_THRESHOLD,
    device_rerouting_adaptation_steps: int = DEVICE_REROUTING_ADAPTATION_STEPS,
    device_rerouting_adaptation_interval: int = DEVICE_REROUTING_ADAPTATION_INTERVAL,
    fcd_output_attributes: str = FCD_OUTPUT_ATTRIBUTES,
) -> None:
    """
    Create a sumo configuration file.

    Args:
        network_path (Path): Path to the network file.
        trips_path (Path): Path to the trips file.
        poly_path (Path): Path to the poly file.
        gui_settings_path (Path): Path to the gui settings file.
        fcd_xml_path (Path): Path to the fcd XML file.
        sumocfg_path (Path): Path to the sumocfg file.
        tls_actuated_jam_threshold (int): Jam threshold for actuated traffic lights.
        device_rerouting_adaptation_steps (int): Number of adaptation steps for device rerouting.
        device_rerouting_adaptation_interval (int): Interval for device rerouting adaptation.
        fcd_output_attributes (str): Attributes for the fcd output.

    Raises:
        FileNotFoundError: If the network, trips, poly, or gui settings file does not exist.
    """
    _validate_path(network_path, "Network file")
    _validate_path(poly_path, "Poly file")
    _validate_path(gui_settings_path, "GUI settings file")

    command = [
        "sumo",
        "--net-file",
        str(network_path),
        "--route-files",
        str(trips_path),
        "--additional-files",
        str(poly_path),
        "--tls.actuated.jam-threshold",
        str(tls_actuated_jam_threshold),
        "--device.rerouting.adaptation-steps",
        str(device_rerouting_adaptation_steps),
        "--device.rerouting.adaptation-interval",
        str(device_rerouting_adaptation_interval),
        "--gui-settings-file",
        str(gui_settings_path),
        "--fcd-output",
        str(fcd_xml_path),
        "--fcd-output.attributes",
        fcd_output_attributes,
        "--save-configuration",
        str(sumocfg_path),
        "--ignore-route-errors",
        "--verbose",
        "--duration-log.statistics",
        "--no-step-log",
    ]
    name = "sumo"
    _execute_command(command, name)


def generate_random_trips(
    network_path: Path,
    trips_path: Path,
    traffic_generation_periods: list[float],
    random_seed: int,
    start_time: int = START_TIME,
    end_time: int = END_TIME,
) -> None:
    """
    Generate random trips using the sumo randomTrips tool.

    Args:
        network_path (Path): Path to the network file.
        trips_path (Path): Path to the trips file.
        traffic_generation_periods (list[float]): List of traffic generation periods.
        random_seed (int): Random seed.
        start_time (int): Start time for trip generation in seconds.
        end_time (int): End time for trip generation in seconds.

    Raises:
        FileNotFoundError: If the network file does not exist.
    """
    _validate_path(network_path, "Network file")

    traffic_generation_periods_str = ",".join(str(v) for v in traffic_generation_periods)
    command = [
        "python",
        str(RANDOM_TRIPS),
        "--net-file",
        str(network_path),
        "--begin",
        str(start_time),
        "--end",
        str(end_time),
        "--period",
        traffic_generation_periods_str,
        "--output-trip-file",
        str(trips_path),
        "--seed",
        str(random_seed),
        "--random-departpos",
        "--random-arrivalpos",
        "--validate",
    ]
    name = "randomTrips"
    _execute_command(command, name)

    routes_temp_path = Path(ROUTES_TEMP_FILENAME)
    routes_temp_path.unlink(missing_ok=True)


def simulate_scenario(sumocfg_path: Path) -> None:
    """
    Run a sumo simulation.

    Args:
        sumocfg_path (Path): Path to the sumocfg file.

    Raises:
        FileNotFoundError: If the sumocfg file does not exist.
    """
    _validate_path(sumocfg_path, "Sumocfg file")

    command = [
        "sumo",
        "--configuration-file",
        str(sumocfg_path),
        "--device.friction.probability",
        str(DEVICE_FRICTION_PROBABILITY),
    ]
    name = "sumo"
    _execute_command(command, name)


def convert_fcd_xml_to_csv(fcd_xml_path: Path) -> None:
    """
    Convert the FCD XML file to CSV and delete the original.

    Args:
        fcd_xml_path (Path): Path to the FCD XML file.

    Raises:
        FileNotFoundError: If the FCD XML file does not exist.
    """
    _validate_path(fcd_xml_path, "FCD XML file")

    command = [
        "python",
        str(XML2CSV),
        str(fcd_xml_path),
    ]
    name = "xml2csv"
    _execute_command(command, name)

    fcd_xml_path.unlink(missing_ok=True)
