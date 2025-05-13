from pathlib import Path

from config import (
    DATASET_SPECS,
    FIXED_FLOWS_FILE,
    FIXED_ROUTES_ALT_FILE,
    FIXED_ROUTES_FILE,
    NETWORK,
)
from generation import (
    convert_xml_to_csv,
    edit_network,
    generate_fixed_routes,
    generate_network,
    generate_random_trips,
    simulate_scenario,
    update_trip_ids,
    update_vehicle_types,
)
from preprocessing import (
    aggregate_fcd,
    parse_fcd_output,
    preprocess_fcd,
    report_fcd_stats,
)
from visualization import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_average_speed_and_vehicle_count_per_second,
    plot_speed_histogram,
)


def generate_dataset(
    dataset_id: str,
    network: Path,
    trips_file: Path,
    traffic_generation_periods: list[int],
    seed: int,
    vehicle_type: str,
    config: Path,
    fcd_output: Path,
    fixed_routes_file: Path | None = None,
    gui: bool = False,
    convert: bool = False,
    delete_original: bool = False,
) -> None:
    """
    Generate a dataset based on the provided parameters.

    Args:
        dataset_id (str): Dataset ID to identify the dataset.
        network (Path): Path to the network file.
        trips_file (Path): Path to the output trips file.
        traffic_generation_periods (list[int]): List of traffic generation periods.
        seed (int): Random seed for trip generation.
        vehicle_type (str): Vehicle type to set in the files.
        config (Path): Path to the SUMO configuration file.
        fcd_output (Path): Path to the XML FCD output file.
        fixed_routes_file (Path | None): Path to the fixed routes file to be updated(optional).
        gui (bool): Flag for running the simulation in GUI mode.
        convert (bool): Flag for converting the FCD output to a CSV file.
        delete_original (bool): Flag for deleting the original XML file after conversion.
    """
    generate_random_trips(
        network=network,
        trips_file=trips_file,
        traffic_generation_periods=traffic_generation_periods,
        seed=seed,
    )
    update_trip_ids(trips_file=trips_file)
    update_vehicle_types(trips_file=trips_file, vehicle_type=vehicle_type, fixed_routes_file=fixed_routes_file)

    simulate_scenario(config=config, gui=gui)

    df_fcd_raw = parse_fcd_output(fcd_output=fcd_output)
    df_fcd = preprocess_fcd(df_fcd=df_fcd_raw)
    report_fcd_stats(df_fcd=df_fcd)
    df_fcd_per_second, df_fcd_per_hour = aggregate_fcd(df_fcd=df_fcd)
    if convert:
        convert_xml_to_csv(xml_file=fcd_output, delete_original=delete_original)

    plot_speed_histogram(df_speed_kmh=df_fcd_per_second["speed_kmh"], dataset_id=dataset_id)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_fcd_per_second["vehicle_count"],
        dataset_id=dataset_id,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=traffic_generation_periods,
        dataset_id=dataset_id,
    )


def main():
    if not NETWORK.exists():
        print("Generating network...")
        generate_network()

    if not FIXED_FLOWS_FILE.exists():
        print("Editing network...")
        edit_network(network=NETWORK)

    if not FIXED_ROUTES_FILE.exists():
        print("Generating fixed flows...")
        generate_fixed_routes(
            network=NETWORK,
            fixed_flows_file=FIXED_FLOWS_FILE,
            fixed_routes_file=FIXED_ROUTES_FILE,
            fixed_routes_alt_file=FIXED_ROUTES_ALT_FILE,
        )

    for spec in DATASET_SPECS:
        name = spec["name"]
        print(f"Generating {name} train dataset...")
        generate_dataset(**spec["train"])
        print(f"Generating {name} test dataset...")
        generate_dataset(**spec["test"])


if __name__ == "__main__":
    main()
