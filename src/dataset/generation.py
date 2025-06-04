import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from config import DUAROUTER, OSM_WEB_WIZARD, RANDOM_TRIPS, XML2CSV

from common.logger import log_subprocess_result, logger


def generate_network() -> None:
    """
    Generate a network using the SUMO osmWebWizard tool.

    Raises:
        Exception: If the network generation fails.
    """
    command = [
        "python",
        str(OSM_WEB_WIZARD),
    ]

    try:
        process = subprocess.Popen(command)

        command_str = " ".join(str(arg) for arg in command)
        logger.info("Executing osmWebWizard")
        logger.info(f"Command: {command_str}")
        logger.info("Generate the network in the osmWebWizard window")
        input("When finished, close the osmWebWizard window and press Enter")

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        logger.info("Network generation completed")

    except Exception as e:
        logger.error(f"Failed to generate network: {e}")
        raise


def edit_network(network: Path) -> None:
    """
    Edit the network file using the SUMO netedit tool.

    Args:
        network (Path): Path to the network file to be edited.

    Raises:
        FileNotFoundError: If the network file does not exist.
        Exception: If the network editing fails.
    """
    if not network.exists():
        error_msg = f"Network file not found: {network}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    command = [
        "netedit",
        str(network),
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        log_subprocess_result("netedit", logger, command, result)

        logger.info(f"Network editing completed: {network}")

    except Exception as e:
        logger.error(f"Failed to edit network: {e}")
        raise


def generate_fixed_routes(
    network: Path, fixed_flows_file: Path, fixed_routes_file: Path, fixed_routes_alt_file: Path
) -> None:
    """
    Generate fixed routes using the SUMO duarouter tool.

    Args:
        network (Path): Path to the network file.
        fixed_flows_file (Path): Path to the fixed flows file.
        fixed_routes_file (Path): Path to the output fixed routes file.
        fixed_routes_alt_file (Path): Path to the alternative output fixed routes file.

    Raises:
        FileNotFoundError: If the network, fixed flows, or fixed routes file does not exist.
        Exception: If the fixed routes generation fails.
    """
    if not network.exists():
        error_msg = f"Network file not found: {network}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    if not fixed_flows_file.exists():
        error_msg = f"Fixed flows file not found: {fixed_flows_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    command = [
        str(DUAROUTER),
        "--net-file",
        str(network),
        "--route-files",
        str(fixed_flows_file),
        "--output-file",
        str(fixed_routes_file),
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        log_subprocess_result("duarouter", logger, command, result)

    except Exception as e:
        logger.error(f"Failed to generate fixed routes: {e}")
        raise

    if fixed_routes_file.exists() and fixed_routes_alt_file.exists():
        fixed_routes_alt_file.unlink()
        logger.info(f"Fixed routes generated successfully: {fixed_routes_file}")
    else:
        error_msg = f"Fixed routes file not found: {fixed_routes_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def generate_random_trips(
    network: Path,
    trips_file: Path,
    traffic_generation_periods: list[int],
    seed: int = 42,
    start_time: int = 0,
    end_time: int = 36000,
) -> None:
    """
    Generate random trips using the SUMO randomTrips tool.

    Args:
        network (Path): Path to the network file.
        trips_file (Path): Path to the output trips file.
        traffic_generation_periods (list[int]): List of traffic generation periods.
        seed (int): Random seed for trip generation.
        start_time (int): Start time for trip generation in seconds.
        end_time (int): End time for trip generation in seconds.

    Raises:
        FileNotFoundError: If the network, or trips file does not exist.
        Exception: If the random trips generation fails.
    """
    if not network.exists():
        error_msg = f"Network file not found: {network}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    traffic_generation_periods_str = ",".join(str(v) for v in traffic_generation_periods)
    routes_temp_file = Path.cwd() / "routes.rou.xml"

    command = [
        "python",
        str(RANDOM_TRIPS),
        "--net-file",
        str(network),
        "--begin",
        str(start_time),
        "--end",
        str(end_time),
        "--period",
        traffic_generation_periods_str,
        "-o",
        str(trips_file),
        "--seed",
        str(seed),
        "--validate",
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        log_subprocess_result("randomTrips", logger, command, result)

    except Exception as e:
        logger.error(f"Failed to generate random trips: {e}")
        raise

    if trips_file.exists() and routes_temp_file.exists():
        routes_temp_file.unlink()
        logger.info(f"Random trips generated successfully: {trips_file}")
    else:
        error_msg = f"Random trips file not found: {trips_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)


def update_trip_ids(trips_file: Path) -> None:
    """
    Update trip IDs in the given trips file to ensure they are unique and sequential.

    Args:
        trips_file (Path): Path to the trips file to be updated.

    Raises:
        FileNotFoundError: If the trips file does not exist.
        Exception: If the trip IDs update fails.
    """
    if not trips_file.exists():
        logger.error(f"Trips file not found: {trips_file}")
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    try:
        tree = ET.parse(trips_file)
        root = tree.getroot()

        trip_id = 0
        for trip in root.findall("trip"):
            trip.set("id", str(trip_id))
            trip_id += 1

        tree.write(trips_file)
        logger.info(f"Updated a total of {trip_id} trip IDs in {trips_file}")

    except Exception as e:
        logger.error(f"Failed to update trip IDs: {e}")
        raise


def update_vehicle_types(trips_file: Path, vehicle_type: str = "car", fixed_routes_file: Path | None = None) -> None:
    """
    Update vehicle types in the given trips file and, if provided, in fixed routes file.

    Args:
        trips_file (Path): Path to the trips file to be updated.
        vehicle_type (str): Vehicle type to set in the files.
        fixed_routes_file (Path | None): Path to the fixed routes file to be updated (optional).

    Raises:
        FileNotFoundError: If the trips, or fixed routes file does not exist.
        Exception: If the vehicle types update fails.
    """
    if not trips_file.exists():
        logger.error(f"Trips file not found: {trips_file}")
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    try:
        tree = ET.parse(trips_file)
        root = tree.getroot()

        trip_count = 0
        for trip in root.findall("trip"):
            trip.set("type", vehicle_type)
            trip_count += 1

        tree.write(trips_file)
        logger.info(f"Vehicle types set to '{vehicle_type}' for {trip_count} trips in {trips_file}")

        if not fixed_routes_file:
            return

        if not fixed_routes_file.exists():
            logger.error(f"Fixed routes file not found: {fixed_routes_file}")
            raise FileNotFoundError(f"Fixed routes file not found: {fixed_routes_file}")

        tree = ET.parse(fixed_routes_file)
        root = tree.getroot()

        vehicle_count = 0
        for vehicle in root.findall("vehicle"):
            vehicle.set("type", vehicle_type)
            vehicle_count += 1

        tree.write(fixed_routes_file)
        logger.info(f"Vehicle types set to '{vehicle_type}' for {vehicle_count} vehicles in {fixed_routes_file}")

    except Exception as e:
        logger.error(f"Failed to update vehicle types: {e}")
        raise


def simulate_scenario(config: Path, gui: bool = False) -> None:
    """
    Run a SUMO simulation using the provided configuration file.

    Args:
        config (Path): Path to the SUMO configuration file.
        gui (bool): Flag for running the simulation in GUI mode.

    Raises:
        FileNotFoundError: If the simulation configuration file does not exist.
        Exception: If the simulation fails.
    """
    if not config.exists():
        error_msg = f"Simulation config file not found: {config}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    command = [
        "sumo-gui" if gui else "sumo",
        "-c",
        str(config),
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        log_subprocess_result("sumo", logger, command, result)

        logger.info(f"Simulation completed: {config}")

    except Exception as e:
        logger.error(f"Failed to simulate scenario: {e}")
        raise


def convert_xml_to_csv(xml_file: Path, delete_original: bool = False) -> None:
    """
    Convert an XML file to a CSV file and delete the original XML file if specified.

    Args:
        xml_file (Path): Path to the XML file to be converted.
        delete_original (bool): Flag for deleting the original XML file after conversion.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        Exception: If the XML to CSV conversion fails.
    """
    if not xml_file.exists():
        error_msg = f"XML file not found: {xml_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    command = [
        "python",
        str(XML2CSV),
        str(xml_file),
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        log_subprocess_result("xml2csv", logger, command, result)

    except Exception as e:
        logger.error(f"Failed to convert XML to CSV: {e}")
        raise

    csv_file = xml_file.with_suffix(".csv")
    if csv_file.exists():
        if delete_original:
            xml_file.unlink()
            logger.info(f"Converted {xml_file} to {csv_file} and deleted original")
        else:
            logger.info(f"Converted {xml_file} to {csv_file}")
    else:
        error_msg = f"CSV file not found: {csv_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
