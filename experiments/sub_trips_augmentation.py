from thesis.common.logger import setup_logger
from thesis.eta.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.eta.experiment import initialize_experiment, save_model, save_results
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model
from thesis.eta.specs import build_scenario_specs


def main() -> None:
    experiment_name = "sub_trips_augmentation"
    artifacts_dir, logs_dir, _, results_dir = initialize_experiment(experiment_name)
    logger = setup_logger(experiment_name, logs_dir)

    scenario_specs = build_scenario_specs()

    for spec in scenario_specs.values():
        logger.info(f"Starting scenario {spec.scenario_name}")

        fcd_train = load_fcd_dataset(spec.train_path)
        fcd_test = load_fcd_dataset(spec.test_path)
        fcd_train = preprocess_fcd_dataset(fcd_train)
        fcd_test = preprocess_fcd_dataset(fcd_test)
        trips_train = generate_trips(fcd_train, augmentation_rate=0.01)
        trips_test = generate_trips(fcd_test)

        X_train, y_train = split_features_and_target(trips_train)
        X_test, y_test = split_features_and_target(trips_test)

        models = get_baseline_models()
        results = {}

        for model_name, model in models.items():
            training_results = train_model(model, model_name, X_train, y_train)
            predictions, prediction_results = make_predictions(model, model_name, X_test)
            evaluation_results = evaluate_predictions(y_test, predictions, model_name)

            results[model_name] = {**training_results, **prediction_results, **evaluation_results}
            save_model(model, model_name, spec.scenario_name, artifacts_dir)

        save_results(results, spec.scenario_name, results_dir)


if __name__ == "__main__":
    main()
