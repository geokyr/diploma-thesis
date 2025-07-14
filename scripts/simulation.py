from thesis.common.logger import setup_logger
from thesis.simulation.config import LOGS_DIR
from thesis.simulation.generation import (
    convert_xml_to_csv_and_move,
    generate_base_network,
    generate_rain_network,
    generate_random_trips,
    simulate_scenario,
)
from thesis.simulation.specs import build_dataset_specs


def main():
    logger = setup_logger("simulation", LOGS_DIR)
    logger.info("Starting simulation process")

    generate_base_network()
    generate_rain_network()

    dataset_specs = build_dataset_specs()

    for spec in dataset_specs.values():
        logger.info(f"Generating {spec.dataset_name} dataset")

        generate_random_trips(
            trips_file=spec.trips_file,
            traffic_generation_periods=spec.traffic_generation_periods,
            network=spec.network_file,
            seed=spec.seed,
        )
        simulate_scenario(config=spec.config)
        convert_xml_to_csv_and_move(xml_file=spec.fcd_output_xml)
        convert_xml_to_csv_and_move(xml_file=spec.emission_output_xml)

        logger.info(f"Generated {spec.dataset_name} dataset")
    logger.info("Completed simulation process")


if __name__ == "__main__":
    main()
