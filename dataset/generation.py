import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from config import DUAROUTER, OSM_WEB_WIZARD, RANDOM_TRIPS, XML2CSV


def generate_network() -> None:
    """
    Generate a network using the SUMO osmWebWizard tool.
    """
    command = [
        "python",
        str(OSM_WEB_WIZARD),
    ]

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from osmWebWizard:")
        print(result.stderr)

    print("Network generated successfully.")


def edit_network(network: Path) -> None:
    """
    Edit the network file using the SUMO netedit tool.

    Args:
        network (Path): Path to the network file to be edited.

    Raises:
        FileNotFoundError: If the network file does not exist.
    """
    if not network.exists():
        raise FileNotFoundError(f"Network file not found: {network}")

    command = [
        "netedit",
        str(network),
    ]

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from netedit:")
        print(result.stderr)

    print(f"Network edited successfully: {network}")


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
        FileNotFoundError: If the network or fixed flows file does not exist.
        RuntimeError: If the fixed routes file generation fails.
    """
    if not network.exists():
        raise FileNotFoundError(f"Network file not found: {network}")

    if not fixed_flows_file.exists():
        raise FileNotFoundError(f"Fixed flows file not found: {fixed_flows_file}")

    command = [
        str(DUAROUTER),
        "--net-file",
        str(network),
        "--route-files",
        str(fixed_flows_file),
        "--output-file",
        str(fixed_routes_file),
    ]

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from duarouter:")
        print(result.stderr)

    if fixed_routes_file.exists() and fixed_routes_alt_file.exists():
        fixed_routes_alt_file.unlink()
        print(f"Fixed routes generated successfully: {fixed_routes_file}")
    else:
        raise RuntimeError(f"Failed to generate fixed routes: {fixed_routes_file}")


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
        FileNotFoundError: If the network file does not exist.
        RuntimeError: If the random trips generation fails.
    """
    if not network.exists():
        raise FileNotFoundError(f"Network file not found: {network}")

    traffic_generation_periods_str = ",".join(str(v) for v in traffic_generation_periods)
    routes_temp_file = Path(__file__).parent / "routes.rou.xml"

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

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from randomTrips:")
        print(result.stderr)

    if trips_file.exists() and routes_temp_file.exists():
        routes_temp_file.unlink()
        print(f"Random trips generated successfully: {trips_file}")
    else:
        raise RuntimeError(f"Failed to generate random trips: {trips_file}")


def update_trip_ids(trips_file: Path) -> None:
    """
    Update trip IDs in the given trips file to ensure they are unique and sequential.

    Args:
        trips_file (Path): Path to the trips file to be updated.

    Raises:
        FileNotFoundError: If the trips file does not exist.
    """
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    tree = ET.parse(trips_file)
    root = tree.getroot()

    trip_id = 0
    for trip in root.findall("trip"):
        trip.set("id", str(trip_id))
        trip_id += 1

    tree.write(trips_file)
    print(f"Updated a total of {trip_id} trip IDs in {trips_file}")


def update_vehicle_types(trips_file: Path, vehicle_type: str = "car", fixed_routes_file: Path | None = None) -> None:
    """
    Update vehicle types in the given trips file and, if provided, in fixed routes file.

    Args:
        trips_file (Path): Path to the trips file to be updated.
        vehicle_type (str): Vehicle type to set in the files.
        fixed_routes_file (Path | None): Path to the fixed routes file to be updated (optional).

    Raises:
        FileNotFoundError: If the trips file or fixed routes file does not exist.
    """
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_file}")

    tree = ET.parse(trips_file)
    root = tree.getroot()

    for trip in root.findall("trip"):
        trip.set("type", vehicle_type)

    tree.write(trips_file)
    print(f"Vehicle types set to '{vehicle_type}' in {trips_file}")

    if not fixed_routes_file:
        return

    if not fixed_routes_file.exists():
        raise FileNotFoundError(f"Fixed routes file not found: {fixed_routes_file}")

    tree = ET.parse(fixed_routes_file)
    root = tree.getroot()

    for vehicle in root.findall("vehicle"):
        vehicle.set("type", vehicle_type)

    tree.write(fixed_routes_file)
    print(f"Vehicle types set to '{vehicle_type}' in {fixed_routes_file}")


def simulate_scenario(config: Path, gui: bool = False) -> None:
    """
    Run a SUMO simulation using the provided configuration file.

    Args:
        config (Path): Path to the SUMO configuration file.
        gui (bool): Flag for running the simulation in GUI mode.

    Raises:
        FileNotFoundError: If the simulation configuration file does not exist.
    """
    if not config.exists():
        raise FileNotFoundError(f"Simulation config file not found: {config}")

    command = [
        "sumo-gui" if gui else "sumo",
        "-c",
        str(config),
    ]

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from sumo:")
        print(result.stderr)

    print(f"Simulation completed successfully: {config}")


def convert_xml_to_csv(xml_file: Path, delete_original: bool = False) -> None:
    """
    Convert an XML file to a CSV file and delete the original XML file if specified.

    Args:
        xml_file (Path): Path to the XML file to be converted.
        delete_original (bool): Flag for deleting the original XML file after conversion.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        RuntimeError: If the conversion fails or the output CSV file is missing.
    """
    if not xml_file.exists():
        raise FileNotFoundError(f"XML file not found: {xml_file}")

    command = [
        "python",
        str(XML2CSV),
        str(xml_file),
    ]

    print("Executing:", " ".join(command))
    result = subprocess.run(command, capture_output=True, check=True, text=True)

    if result.stderr:
        print("Warnings/Errors from xml2csv:")
        print(result.stderr)

    csv_file = xml_file.with_suffix(".csv")
    if csv_file.exists():
        if delete_original:
            xml_file.unlink()
        print(f"Converted {xml_file} to {csv_file}")
    else:
        raise RuntimeError(f"Failed to convert {xml_file}, missing output file: {csv_file}")
