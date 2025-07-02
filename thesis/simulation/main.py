from thesis.common.logger import setup_logger
from thesis.simulation.config import LOGS_DIR
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
from thesis.simulation.utils import build_dataset_specs


def main():
    logger = setup_logger("simulation", LOGS_DIR)
    logger.info("Starting simulation process")

    generate_network()
    edit_network()
    generate_fixed_routes()

    dataset_specs = build_dataset_specs()
    for spec in dataset_specs.values():
        logger.info(f"Generating {spec.dataset_name} dataset")

        generate_random_trips(
            trips_file=spec.trips_file,
            traffic_generation_periods=spec.traffic_generation_periods,
            seed=spec.seed,
        )
        update_trip_ids(trips_file=spec.trips_file)
        update_vehicle_types(
            trips_file=spec.trips_file,
            vehicle_type=spec.vehicle_type,
            fixed_routes_file=spec.fixed_routes_file,
        )

        simulate_scenario(config=spec.config)

        for xml_file in spec.xml_files:
            convert_xml_to_csv(xml_file=xml_file)

        logger.info(f"Completed dataset generation for {spec.dataset_name} dataset")

    logger.info("Completed simulation process")


if __name__ == "__main__":
    main()
