from thesis.logger import setup_logger
from thesis.simulation.config import DATASET_SPECS, LOGS_DIR
from thesis.simulation.generation import (
    convert_xml_to_csv,
    edit_network,
    generate_fixed_routes,
    generate_network,
    generate_random_trips,
    simulate_scenario,
    update_trip_ids,
    update_vehicle_types,
)


def main():
    logger = setup_logger("simulation", LOGS_DIR)
    logger.info("Starting simulation process")

    generate_network()
    edit_network()
    generate_fixed_routes()

    for dataset_name, dataset_spec in DATASET_SPECS.items():
        logger.info(f"Generating {dataset_name} dataset")

        trips_file = dataset_spec["trips_file"]
        traffic_generation_periods = dataset_spec["traffic_generation_periods"]
        seed = dataset_spec["seed"]
        config = dataset_spec["config"]
        fcd_output = dataset_spec["fcd_output"]
        fixed_routes_file = dataset_spec["fixed_routes_file"]
        vehicle_type = dataset_spec["vehicle_type"]

        generate_random_trips(
            trips_file=trips_file,
            traffic_generation_periods=traffic_generation_periods,
            seed=seed,
        )
        update_trip_ids(trips_file=trips_file)
        update_vehicle_types(trips_file=trips_file, vehicle_type=vehicle_type, fixed_routes_file=fixed_routes_file)

        simulate_scenario(config=config)
        convert_xml_to_csv(xml_file=fcd_output)

        logger.info(f"Completed dataset generation for {dataset_name} dataset")

    logger.info("Completed dataset generation process")


if __name__ == "__main__":
    main()
