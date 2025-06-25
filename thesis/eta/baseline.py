import os

from thesis.common.config import TYPE_TEST, TYPE_TRAIN
from thesis.common.logger import setup_logger
from thesis.eta.config import SCENARIOS_SPECS
from thesis.eta.data import load_fcd_dataset, prepare_baseline_trips, preprocess_fcd_dataset
from thesis.eta.eda import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_speed_histogram,
    plot_trips_distances_distribution,
    plot_trips_durations_distribution,
    report_fcd_statistics,
    report_trips_statistics,
)
from thesis.eta.experiment import (
    initialize_experiment,
    save_experiment_results,
    save_model,
    save_scenario_results,
)
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import train_and_evaluate_model


def main() -> None:
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    EXPERIMENT_NAME = "baseline"
    artifacts_dir, logs_dir, plots_dir, results_dir = initialize_experiment(EXPERIMENT_NAME)
    logger = setup_logger(EXPERIMENT_NAME, logs_dir)

    logger.info(f"Starting experiment {EXPERIMENT_NAME}")

    experiment_results = {}
    for scenario_name, train_path, test_path in SCENARIOS_SPECS:
        logger.info(f"Starting scenario {scenario_name}")

        train_df = load_fcd_dataset(train_path)
        test_df = load_fcd_dataset(test_path)
        train_df = preprocess_fcd_dataset(train_df)
        test_df = preprocess_fcd_dataset(test_df)
        train_trips = prepare_baseline_trips(train_df)
        test_trips = prepare_baseline_trips(test_df)
        X_train, y_train = split_features_and_target(train_trips)
        X_test, y_test = split_features_and_target(test_trips)

        TRAIN_DATASET_ID = f"{scenario_name}-{TYPE_TRAIN}"
        TEST_DATASET_ID = f"{scenario_name}-{TYPE_TEST}"

        report_fcd_statistics(train_df, TRAIN_DATASET_ID)
        report_fcd_statistics(test_df, TEST_DATASET_ID)
        report_trips_statistics(train_trips, TRAIN_DATASET_ID)
        report_trips_statistics(test_trips, TEST_DATASET_ID)

        plot_speed_histogram(train_df, TRAIN_DATASET_ID, plots_dir)
        plot_speed_histogram(test_df, TEST_DATASET_ID, plots_dir)
        plot_average_speed_and_traffic_generation_period_per_hour(train_df, TRAIN_DATASET_ID, plots_dir)
        plot_average_speed_and_traffic_generation_period_per_hour(test_df, TEST_DATASET_ID, plots_dir)
        plot_trips_distances_distribution(train_trips, TRAIN_DATASET_ID, plots_dir)
        plot_trips_distances_distribution(test_trips, TEST_DATASET_ID, plots_dir)
        plot_trips_durations_distribution(train_trips, TRAIN_DATASET_ID, plots_dir)
        plot_trips_durations_distribution(test_trips, TEST_DATASET_ID, plots_dir)

        models = get_baseline_models()
        scenario_results = {}
        for model_name, model in models.items():
            model_results = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
            scenario_results[model_name] = model_results
            save_model(model, model_name, scenario_name, artifacts_dir)

        experiment_results[scenario_name] = scenario_results
        save_scenario_results(scenario_results, scenario_name, results_dir)
        logger.info(f"Completed scenario {scenario_name}")

    save_experiment_results(experiment_results, EXPERIMENT_NAME, results_dir)
    logger.info(f"Completed experiment {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
