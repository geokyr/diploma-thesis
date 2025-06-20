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
from thesis.logger import setup_logging


def main():
    EXPERIMENT_NAME = "dataset"
    logger = setup_logging(EXPERIMENT_NAME)

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

    for dataset_name, dataset_spec in DATASET_SPECS.items():
        logger.info(f"Generating {dataset_name} dataset")

        dataset_id = dataset_spec["dataset_id"]
        trips_file = dataset_spec["trips_file"]
        traffic_generation_periods = dataset_spec["traffic_generation_periods"]
        seed = dataset_spec["seed"]
        config = dataset_spec["config"]
        fcd_output = dataset_spec["fcd_output"]
        fixed_routes_file = dataset_spec["fixed_routes_file"]
        vehicle_type = dataset_spec["vehicle_type"]
        network = dataset_spec["network"]
        gui = dataset_spec["gui"]
        convert = dataset_spec["convert"]
        delete_original = dataset_spec["delete_original"]

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

        logger.info(f"Completed dataset generation for {dataset_id} dataset")

    logger.info("Completed dataset generation process")


if __name__ == "__main__":
    main()
