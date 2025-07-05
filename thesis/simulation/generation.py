import gzip
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from thesis.common.config import DATA_DIR
from thesis.simulation.config import (
    CLOSURE_ADDITIONAL_FILE,
    NETWORK,
    OSM_WEB_WIZARD,
    RAIN_NETWORK,
    RANDOM_TRIPS,
    VEHICLE_CAR,
    XML2CSV,
)

logger = logging.getLogger(__name__)


def generate_network() -> None:
    """
    Generate a network using the SUMO osmWebWizard tool.

    Raises:
        Exception: If the network generation fails.
    """

    if NETWORK.exists():
        logger.info("Network file already exists, skipping network generation")
        return

    command = [
        "python",
        str(OSM_WEB_WIZARD),
    ]

    try:
        command_str = " ".join(str(arg) for arg in command)
        logger.info(f"Executing: {command_str}")

        process = subprocess.Popen(command)

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


def generate_rain_network(friction: float = 0.4) -> None:
    """
    Generate a rain network file with friction values by modifying the existing network XML.

    Args:
        friction (float): Friction value to apply to all edges.

    Raises:
        FileNotFoundError: If the base network file does not exist.
        Exception: If the network generation fails.
    """
    if RAIN_NETWORK.exists():
        logger.info("Rain network file already exists, skipping generation")
        return

    if not NETWORK.exists():
        error_msg = f"Base network file not found: {NETWORK}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        logger.info(f"Reading network from {NETWORK}")

        with gzip.open(NETWORK, "rb") as f_in:
            tree = ET.parse(f_in)
            root = tree.getroot()

        lanes_updated = 0
        for lane in root.findall(".//lane"):
            lane.set("friction", str(friction))
            lanes_updated += 1

        with gzip.open(RAIN_NETWORK, "wb") as f_out:
            tree.write(f_out, xml_declaration=True, encoding="UTF-8")

        logger.info(f"Generated rain network with {friction} friction on {lanes_updated} lanes")

    except Exception as e:
        logger.error(f"Failed to generate rain network: {e}")
        raise


def edit_network() -> None:
    """
    Edit the network file using the SUMO netedit tool.

    Raises:
        FileNotFoundError: If the network file does not exist.
        Exception: If the network editing fails.
    """
    if CLOSURE_ADDITIONAL_FILE.exists():
        logger.info("Closure additional file already exists, skipping network editing")
        return

    if not NETWORK.exists():
        error_msg = f"Network file not found: {NETWORK}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    command = [
        "netedit",
        str(NETWORK),
    ]

    try:
        command_str = " ".join(str(arg) for arg in command)
        logger.info(f"Executing: {command_str}")

        process = subprocess.Popen(command)

        logger.info("Edit the network in the netedit window")
        input("When finished, close the netedit window and press Enter")

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        logger.info(f"Network editing completed: {NETWORK}")

    except Exception as e:
        logger.error(f"Failed to edit network: {e}")
        raise


def generate_random_trips(
    trips_file: Path,
    traffic_generation_periods: list[float],
    seed: int = 42,
    start_time: int = 0,
    end_time: int = 36000,
) -> None:
    """
    Generate random trips using the SUMO randomTrips tool.

    Args:
        trips_file (Path): Path to the output trips file.
        traffic_generation_periods (list[float]): List of traffic generation periods.
        seed (int): Random seed for trip generation.
        start_time (int): Start time for trip generation in seconds.
        end_time (int): End time for trip generation in seconds.

    Raises:
        FileNotFoundError: If the network, or trips file does not exist.
        subprocess.CalledProcessError: If the random trips generation returns a non-zero exit code.
        Exception: If the random trips generation fails.
    """
    if not NETWORK.exists():
        error_msg = f"Network file not found: {NETWORK}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    traffic_generation_periods_str = ",".join(str(v) for v in traffic_generation_periods)
    routes_temp_file = Path.cwd() / "routes.rou.xml"

    command = [
        "python",
        str(RANDOM_TRIPS),
        "--net-file",
        str(NETWORK),
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
        command_str = " ".join(str(arg) for arg in command)
        logger.info(f"Executing: {command_str}")

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"randomTrips: {line}")

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    except subprocess.CalledProcessError as e:
        logger.error(f"Random trips generation failed with return code {e.returncode}")
        raise
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
        logger.info(f"Updating trip IDs in {trips_file}")

        tree = ET.parse(trips_file)
        root = tree.getroot()

        trip_id = 0
        for trip in root.findall("trip"):
            trip.set("id", str(trip_id))
            trip_id += 1

        tree.write(trips_file)
        logger.info(f"Updated a total of {trip_id} trip IDs")

    except Exception as e:
        logger.error(f"Failed to update trip IDs: {e}")
        raise


def update_vehicle_types(trips_file: Path) -> None:
    """
    Update vehicle types in the given trips file.

    Args:
        trips_file (Path): Path to the trips file to be updated.

    Raises:
        FileNotFoundError: If the trips file does not exist.
        Exception: If the vehicle types update fails.
    """
    if not trips_file.exists():
        logger.error(f"Trips file not found: {trips_file}")
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    try:
        logger.info(f"Updating vehicle types in {trips_file}")

        tree = ET.parse(trips_file)
        root = tree.getroot()

        trip_count = 0
        for trip in root.findall("trip"):
            trip.set("type", VEHICLE_CAR)
            trip_count += 1

        tree.write(trips_file)
        logger.info(f"Vehicle types set to '{VEHICLE_CAR}' for {trip_count} trips")

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
        subprocess.CalledProcessError: If the simulation returns a non-zero exit code.
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
        command_str = " ".join(str(arg) for arg in command)
        logger.info(f"Executing: {command_str}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"SUMO: {line}")

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        logger.info(f"Simulation completed successfully: {config}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Simulation failed with return code {e.returncode}")
        raise
    except Exception as e:
        logger.error(f"Failed to simulate scenario: {e}")
        raise


def convert_xml_to_csv_and_move(xml_file: Path) -> None:
    """
    Convert an XML file to CSV, delete the original XML, and move the CSV to the data directory.

    Args:
        xml_file (Path): Path to the XML file to be converted.

    Raises:
        FileNotFoundError: If the XML file does not exist, or the CSV file does not exist after conversion.
        subprocess.CalledProcessError: If the conversion returns a non-zero exit code.
        Exception: If the conversion or moving fails.
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
        command_str = " ".join(str(arg) for arg in command)
        logger.info(f"Executing: {command_str}")

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
        )

        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(f"xml2csv: {line}")

        return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    except subprocess.CalledProcessError as e:
        logger.error(f"XML to CSV conversion failed with return code {e.returncode}")
        raise
    except Exception as e:
        logger.error(f"Failed to convert XML to CSV: {e}")
        raise

    csv_file = xml_file.with_suffix(".csv")
    if not csv_file.exists():
        error_msg = f"CSV file not found after conversion: {csv_file}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    xml_file.unlink()
    logger.info(f"Converted {xml_file} to {csv_file} and deleted original")

    try:
        destination = DATA_DIR / csv_file.name
        csv_file.rename(destination)
        logger.info(f"Moved {csv_file} to {destination}")

    except Exception as e:
        logger.error(f"Failed to move CSV file: {e}")
        raise
