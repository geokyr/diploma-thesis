import os
import subprocess
from pathlib import Path

from config import DUAROUTER, NETWORK


def generate_fixed_routes(fixed_flows_file: Path, fixed_routes_file: Path, fixed_routes_alt_file: Path) -> None:
    """
    Generate fixed routes using the DUAROUTER tool.

    Args:
        fixed_flows_file (Path): Path to the fixed flows file.
        fixed_routes_file (Path): Path to the output fixed routes file.
        fixed_routes_alt_file (Path): Path to the alternative output fixed routes file.
    """
    command = [
        str(DUAROUTER),
        "--net-file",
        str(NETWORK),
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

    if os.path.exists(fixed_routes_file) and os.path.exists(fixed_routes_alt_file):
        os.remove(fixed_routes_alt_file)
        print("Success:", fixed_routes_file)
    else:
        print("Failed:", fixed_routes_file)


def generate_random_trips():
    pass


def update_trip_ids():
    pass


def update_vehicle_types():
    pass


def simulate_scenario():
    pass


def simulate_gui_scenario():
    pass


def edit_network():
    pass


def generate_network():
    pass


def convert_xml_to_csv():
    pass
