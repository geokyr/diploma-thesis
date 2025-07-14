from thesis.common.config import TYPE_TEST, TYPE_TRAIN
from thesis.common.logger import setup_logger
from thesis.eta.config import SCENARIOS_SPECS
from thesis.eta.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.eta.eda import (
    plot_average_speed_and_traffic_generation_period_per_hour,
    plot_speed_histogram,
    plot_trips_distances_distribution,
    plot_trips_durations_distribution,
    report_fcd_statistics,
    report_trips_statistics,
)
from thesis.eta.experiment import initialize_experiment, save_model, save_results
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model


def main() -> None:
    experiment_name = "baseline"
    artifacts_dir, logs_dir, plots_dir, results_dir = initialize_experiment(experiment_name)
    logger = setup_logger(experiment_name, logs_dir)

    for scenario_name, train_path, test_path in SCENARIOS_SPECS:
        logger.info(f"Starting scenario {scenario_name}")

        fcd_train = load_fcd_dataset(train_path)
        fcd_test = load_fcd_dataset(test_path)
        fcd_train = preprocess_fcd_dataset(fcd_train)
        fcd_test = preprocess_fcd_dataset(fcd_test)
        trips_train = generate_trips(fcd_train)
        trips_test = generate_trips(fcd_test)

        if "-" not in scenario_name:
            train_dataset_id = f"{scenario_name}-{TYPE_TRAIN}"
            test_dataset_id = f"{scenario_name}-{TYPE_TEST}"
            report_fcd_statistics(fcd_train, train_dataset_id)
            report_fcd_statistics(fcd_test, test_dataset_id)
            report_trips_statistics(trips_train, train_dataset_id)
            report_trips_statistics(trips_test, test_dataset_id)
            plot_speed_histogram(fcd_train, train_dataset_id, plots_dir)
            plot_speed_histogram(fcd_test, test_dataset_id, plots_dir)
            plot_average_speed_and_traffic_generation_period_per_hour(fcd_train, train_dataset_id, plots_dir)
            plot_average_speed_and_traffic_generation_period_per_hour(fcd_test, test_dataset_id, plots_dir)
            plot_trips_distances_distribution(trips_train, train_dataset_id, plots_dir)
            plot_trips_distances_distribution(trips_test, test_dataset_id, plots_dir)
            plot_trips_durations_distribution(trips_train, train_dataset_id, plots_dir)
            plot_trips_durations_distribution(trips_test, test_dataset_id, plots_dir)

        X_train, y_train = split_features_and_target(trips_train)
        X_test, y_test = split_features_and_target(trips_test)

        models = get_baseline_models()
        results = {}

        for model_name, model in models.items():
            training_results = train_model(model, model_name, X_train, y_train)
            predictions, prediction_results = make_predictions(model, model_name, X_test)
            evaluation_results = evaluate_predictions(y_test, predictions, model_name)

            results[model_name] = {**training_results, **prediction_results, **evaluation_results}
            save_model(model, model_name, scenario_name, artifacts_dir)

        save_results(results, scenario_name, results_dir)


if __name__ == "__main__":
    main()
