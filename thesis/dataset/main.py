from pathlib import Path

from thesis.common.logger import DATASET_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger
from thesis.dataset.config import (
    DATASET_SPECS,
    FIXED_FLOWS_FILE,
    FIXED_ROUTES_ALT_FILE,
    FIXED_ROUTES_FILE,
    NETWORK,
)
from thesis.dataset.generation import (
    convert_xml_to_csv,
    edit_network,
    generate_fixed_routes,
    generate_network,
    generate_random_trips,
    simulate_scenario,
    update_trip_ids,
    update_vehicle_types,
)
from thesis.dataset.preprocessing import aggregate_fcd, parse_fcd_output, preprocess_fcd, report_fcd_stats
from thesis.dataset.visualization import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_average_speed_and_vehicle_count_per_second,
    plot_speed_histogram,
)

logger = setup_logger(name=DATASET_LOGGER_NAME, log_file=LOG_FILES_CONFIG[DATASET_LOGGER_NAME])


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
        fixed_routes_file (Path | None): Path to the fixed routes file to be updated (optional).
        gui (bool): Flag for running the simulation in GUI mode.
        convert (bool): Flag for converting the FCD output to a CSV file.
        delete_original (bool): Flag for deleting the original XML file after conversion.
    """
    logger.info(f"Starting dataset generation for: {dataset_id}")

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

    plot_speed_histogram(speeds_kmh=df_fcd["speed_kmh"], dataset_id=dataset_id)
    plot_average_speed_and_vehicle_count_per_second(
        seconds=df_fcd_per_second["second"],
        average_speeds_kmh_per_second=df_fcd_per_second["average_speed_kmh"],
        vehicle_counts_per_second=df_fcd_per_second["vehicle_count"],
        dataset_id=dataset_id,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        hours=df_fcd_per_hour["hour"],
        average_speeds_kmh_per_hour=df_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=traffic_generation_periods,
        dataset_id=dataset_id,
    )

    logger.info(f"Completed dataset generation for: {dataset_id}")


def main():
    logger.info("Starting dataset generation process")

    if not NETWORK.exists():
        logger.info("Network file not found, generating network")
        generate_network()

    if not FIXED_FLOWS_FILE.exists():
        logger.info("Fixed flows file not found, editing network")
        edit_network(network=NETWORK)

    if not FIXED_ROUTES_FILE.exists():
        logger.info("Fixed routes file not found, generating fixed flows")
        generate_fixed_routes(
            network=NETWORK,
            fixed_flows_file=FIXED_FLOWS_FILE,
            fixed_routes_file=FIXED_ROUTES_FILE,
            fixed_routes_alt_file=FIXED_ROUTES_ALT_FILE,
        )

    for spec in DATASET_SPECS:
        name = spec["name"]
        logger.info(f"Generating {name} train dataset")
        generate_dataset(**spec["train"])
        logger.info(f"Generating {name} test dataset")
        generate_dataset(**spec["test"])

    logger.info("Dataset generation process completed successfully")


if __name__ == "__main__":
    main()
