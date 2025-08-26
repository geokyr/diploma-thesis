from thesis.common.logger import setup_logger
from thesis.simulation.experiment import SimulationExperiment
from thesis.simulation.pipeline import (
    build_network,
    build_rain_network,
    convert_fcd_csv_to_parquet,
    create_configuration_file,
    generate_random_trips,
    get_osm_data,
    run_fcd_exploratory_data_analysis,
    simulate_scenario,
    write_gui_settings_file,
)
from thesis.simulation.scenario import SimulationScenario, SimulationScenarioConfig


def main():
    experiment = SimulationExperiment()
    logger = setup_logger(experiment.name, experiment.logs_dir)

    get_osm_data(simulation_dir=experiment.simulation_dir)
    build_network(simulation_dir=experiment.simulation_dir, osm_data_path=experiment.osm_data_path)
    build_rain_network(network_base_path=experiment.network_base_path, network_rain_path=experiment.network_rain_path)
    write_gui_settings_file(gui_settings_path=experiment.gui_settings_path)

    for scenario in SimulationScenario:
        logger.info(f"Generating {scenario} scenario")
        config = SimulationScenarioConfig(scenario=scenario, simulation_dir=experiment.simulation_dir)

        create_configuration_file(
            network_path=config.network_path,
            trips_path=config.trips_path,
            poly_path=experiment.poly_path,
            gui_settings_path=experiment.gui_settings_path,
            fcd_csv_path=config.fcd_csv_path,
            sumocfg_path=config.sumocfg_path,
        )
        generate_random_trips(
            network_path=config.network_path,
            trips_path=config.trips_path,
            traffic_generation_periods=config.traffic_generation_periods,
            random_seed=config.random_seed,
        )
        simulate_scenario(sumocfg_path=config.sumocfg_path)
        convert_fcd_csv_to_parquet(fcd_csv_path=config.fcd_csv_path)
        run_fcd_exploratory_data_analysis(
            fcd_parquet_path=config.fcd_parquet_path,
            scenario=config.scenario,
            plots_dir=experiment.plots_dir,
            traffic_generation_periods=config.traffic_generation_periods,
        )


if __name__ == "__main__":
    main()
