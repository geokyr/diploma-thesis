from config import (
    BASE_TEST_DATASET_ID,
    BASE_TEST_FCD,
    BASE_TEST_SIMULATION_CONFIG,
    BASE_TEST_TRIPS_FILE,
    BASE_TRAIN_DATASET_ID,
    BASE_TRAIN_FCD,
    BASE_TRAIN_SIMULATION_CONFIG,
    BASE_TRAIN_TRIPS_FILE,
    CLOSURE_TEST_DATASET_ID,
    CLOSURE_TEST_FCD,
    CLOSURE_TEST_SIMULATION_CONFIG,
    CLOSURE_TEST_TRIPS_FILE,
    CLOSURE_TRAIN_DATASET_ID,
    CLOSURE_TRAIN_FCD,
    CLOSURE_TRAIN_SIMULATION_CONFIG,
    CLOSURE_TRAIN_TRIPS_FILE,
    FIXED_FLOWS_FILE,
    FIXED_ROUTES_ALT_FILE,
    FIXED_ROUTES_FILE,
    NETWORK,
    RAIN_TEST_DATASET_ID,
    RAIN_TEST_FCD,
    RAIN_TEST_SIMULATION_CONFIG,
    RAIN_TEST_TRIPS_FILE,
    RAIN_TRAIN_DATASET_ID,
    RAIN_TRAIN_FCD,
    RAIN_TRAIN_SIMULATION_CONFIG,
    RAIN_TRAIN_TRIPS_FILE,
    TEST_SEED,
    TEST_TRAFFIC_GENERATION_PERIODS,
    TRAIN_SEED,
    TRAIN_TRAFFIC_GENERATION_PERIODS,
)
from generation import (
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
)
from visualization import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_average_speed_and_vehicle_count_per_second,
    plot_speed_histogram,
)


def generate_base_train_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=BASE_TRAIN_TRIPS_FILE,
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        seed=TRAIN_SEED,
    )
    update_trip_ids(trips_file=BASE_TRAIN_TRIPS_FILE)
    update_vehicle_types(trips_file=BASE_TRAIN_TRIPS_FILE, vehicle_type="car")

    simulate_scenario(simulation_config=BASE_TRAIN_SIMULATION_CONFIG)

    df_base_train_fcd = parse_fcd_output(fcd_output=BASE_TRAIN_FCD)
    df_base_train_fcd = preprocess_fcd(df_fcd=df_base_train_fcd)
    df_base_train_fcd_per_second, df_base_train_fcd_per_hour = aggregate_fcd(df_fcd=df_base_train_fcd)

    plot_speed_histogram(df_speed_kmh=df_base_train_fcd_per_second["speed_kmh"], dataset_id=BASE_TRAIN_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_base_train_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_base_train_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_base_train_fcd_per_second["vehicle_count"],
        dataset_id=BASE_TRAIN_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_base_train_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_base_train_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        dataset_id=BASE_TRAIN_DATASET_ID,
    )


def generate_base_test_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=BASE_TEST_TRIPS_FILE,
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        seed=TEST_SEED,
    )
    update_trip_ids(trips_file=BASE_TEST_TRIPS_FILE)
    update_vehicle_types(trips_file=BASE_TEST_TRIPS_FILE, fixed_routes_file=FIXED_ROUTES_FILE, vehicle_type="car")

    simulate_scenario(simulation_config=BASE_TEST_SIMULATION_CONFIG)

    df_base_test_fcd = parse_fcd_output(fcd_output=BASE_TEST_FCD)
    df_base_test_fcd = preprocess_fcd(df_fcd=df_base_test_fcd)
    df_base_test_fcd_per_second, df_base_test_fcd_per_hour = aggregate_fcd(df_fcd=df_base_test_fcd)

    plot_speed_histogram(df_speed_kmh=df_base_test_fcd_per_second["speed_kmh"], dataset_id=BASE_TEST_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_base_test_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_base_test_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_base_test_fcd_per_second["vehicle_count"],
        dataset_id=BASE_TEST_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_base_test_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_base_test_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        dataset_id=BASE_TEST_DATASET_ID,
    )


def generate_closure_train_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=CLOSURE_TRAIN_TRIPS_FILE,
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        seed=TRAIN_SEED,
    )
    update_trip_ids(trips_file=CLOSURE_TRAIN_TRIPS_FILE)
    update_vehicle_types(trips_file=CLOSURE_TRAIN_TRIPS_FILE, vehicle_type="car")

    simulate_scenario(simulation_config=CLOSURE_TRAIN_SIMULATION_CONFIG)

    df_closure_train_fcd = parse_fcd_output(fcd_output=CLOSURE_TRAIN_FCD)
    df_closure_train_fcd = preprocess_fcd(df_fcd=df_closure_train_fcd)
    df_closure_train_fcd_per_second, df_closure_train_fcd_per_hour = aggregate_fcd(df_fcd=df_closure_train_fcd)

    plot_speed_histogram(df_speed_kmh=df_closure_train_fcd_per_second["speed_kmh"], dataset_id=CLOSURE_TRAIN_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_closure_train_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_closure_train_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_closure_train_fcd_per_second["vehicle_count"],
        dataset_id=CLOSURE_TRAIN_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_closure_train_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_closure_train_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        dataset_id=CLOSURE_TRAIN_DATASET_ID,
    )


def generate_closure_test_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=CLOSURE_TEST_TRIPS_FILE,
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        seed=TEST_SEED,
    )
    update_trip_ids(trips_file=CLOSURE_TEST_TRIPS_FILE)
    update_vehicle_types(trips_file=CLOSURE_TEST_TRIPS_FILE, fixed_routes_file=FIXED_ROUTES_FILE, vehicle_type="car")

    simulate_scenario(simulation_config=CLOSURE_TEST_SIMULATION_CONFIG)

    df_closure_test_fcd = parse_fcd_output(fcd_output=CLOSURE_TEST_FCD)
    df_closure_test_fcd = preprocess_fcd(df_fcd=df_closure_test_fcd)
    df_closure_test_fcd_per_second, df_closure_test_fcd_per_hour = aggregate_fcd(df_fcd=df_closure_test_fcd)

    plot_speed_histogram(df_speed_kmh=df_closure_test_fcd_per_second["speed_kmh"], dataset_id=CLOSURE_TEST_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_closure_test_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_closure_test_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_closure_test_fcd_per_second["vehicle_count"],
        dataset_id=CLOSURE_TEST_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_closure_test_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_closure_test_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        dataset_id=CLOSURE_TEST_DATASET_ID,
    )


def generate_rain_train_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=RAIN_TRAIN_TRIPS_FILE,
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        seed=TRAIN_SEED,
    )
    update_trip_ids(trips_file=RAIN_TRAIN_TRIPS_FILE)
    update_vehicle_types(trips_file=RAIN_TRAIN_TRIPS_FILE, vehicle_type="car-rain")

    simulate_scenario(simulation_config=RAIN_TRAIN_SIMULATION_CONFIG)

    df_rain_train_fcd = parse_fcd_output(fcd_output=RAIN_TRAIN_FCD)
    df_rain_train_fcd = preprocess_fcd(df_fcd=df_rain_train_fcd)
    df_rain_train_fcd_per_second, df_rain_train_fcd_per_hour = aggregate_fcd(df_fcd=df_rain_train_fcd)

    plot_speed_histogram(df_speed_kmh=df_rain_train_fcd_per_second["speed_kmh"], dataset_id=RAIN_TRAIN_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_rain_train_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_rain_train_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_rain_train_fcd_per_second["vehicle_count"],
        dataset_id=RAIN_TRAIN_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_rain_train_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_rain_train_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TRAIN_TRAFFIC_GENERATION_PERIODS,
        dataset_id=RAIN_TRAIN_DATASET_ID,
    )


def generate_rain_test_dataset():
    generate_random_trips(
        network=NETWORK,
        trips_file=RAIN_TEST_TRIPS_FILE,
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        seed=TEST_SEED,
    )
    update_trip_ids(trips_file=RAIN_TEST_TRIPS_FILE)
    update_vehicle_types(trips_file=RAIN_TEST_TRIPS_FILE, fixed_routes_file=FIXED_ROUTES_FILE, vehicle_type="car-rain")

    simulate_scenario(simulation_config=RAIN_TEST_SIMULATION_CONFIG)

    df_rain_test_fcd = parse_fcd_output(fcd_output=RAIN_TEST_FCD)
    df_rain_test_fcd = preprocess_fcd(df_fcd=df_rain_test_fcd)
    df_rain_test_fcd_per_second, df_rain_test_fcd_per_hour = aggregate_fcd(df_fcd=df_rain_test_fcd)

    plot_speed_histogram(df_speed_kmh=df_rain_test_fcd_per_second["speed_kmh"], dataset_id=RAIN_TEST_DATASET_ID)
    plot_average_speed_and_vehicle_count_per_second(
        df_second=df_rain_test_fcd_per_second["second"],
        df_average_speed_kmh_per_second=df_rain_test_fcd_per_second["average_speed_kmh"],
        df_vehicle_count_per_second=df_rain_test_fcd_per_second["vehicle_count"],
        dataset_id=RAIN_TEST_DATASET_ID,
    )
    plot_average_speed_and_traffic_generation_period_per_hour(
        df_hour=df_rain_test_fcd_per_hour["hour"],
        df_average_speed_kmh_per_hour=df_rain_test_fcd_per_hour["average_speed_kmh"],
        traffic_generation_periods=TEST_TRAFFIC_GENERATION_PERIODS,
        dataset_id=RAIN_TEST_DATASET_ID,
    )


def main():
    print("Generating network...")
    generate_network()

    print("Editing network...")
    edit_network(network=NETWORK)

    print("Generating fixed routes...")
    generate_fixed_routes(
        network=NETWORK,
        fixed_flows_file=FIXED_FLOWS_FILE,
        fixed_routes_file=FIXED_ROUTES_FILE,
        fixed_routes_alt_file=FIXED_ROUTES_ALT_FILE,
    )

    print("Generating base train dataset...")
    generate_base_train_dataset()

    print("Generating base test dataset...")
    generate_base_test_dataset()

    print("Generating closure train dataset...")
    generate_closure_train_dataset()

    print("Generating closure test dataset...")
    generate_closure_test_dataset()

    print("Generating rain train dataset...")
    generate_rain_train_dataset()

    print("Generating rain test dataset...")
    generate_rain_test_dataset()


if __name__ == "__main__":
    main()
