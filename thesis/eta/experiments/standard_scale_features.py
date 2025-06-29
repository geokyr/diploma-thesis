from thesis.common.logger import setup_logger
from thesis.eta.config import SCENARIOS_SPECS
from thesis.eta.data import load_fcd_dataset, prepare_baseline_trips, preprocess_fcd_dataset
from thesis.eta.experiment import initialize_experiment, save_model, save_results
from thesis.eta.features import split_features_and_target, standard_scale_features
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model


def main() -> None:
    experiment_name = "standard_scaler"
    artifacts_dir, logs_dir, _, results_dir = initialize_experiment(experiment_name)
    logger = setup_logger(experiment_name, logs_dir)

    for scenario_name, train_path, test_path in SCENARIOS_SPECS:
        logger.info(f"Starting scenario {scenario_name}")

        fcd_train = load_fcd_dataset(train_path)
        fcd_test = load_fcd_dataset(test_path)
        fcd_train = preprocess_fcd_dataset(fcd_train)
        fcd_test = preprocess_fcd_dataset(fcd_test)
        trips_train = prepare_baseline_trips(fcd_train)
        trips_test = prepare_baseline_trips(fcd_test)

        X_train, y_train = split_features_and_target(trips_train)
        X_test, y_test = split_features_and_target(trips_test)
        X_train_scaled, X_test_scaled = standard_scale_features(X_train, X_test)

        models = get_baseline_models()
        results = {}
        for model_name, model in models.items():
            training_results = train_model(model, model_name, X_train_scaled, y_train)
            predictions, prediction_results = make_predictions(model, model_name, X_test_scaled)
            evaluation_results = evaluate_predictions(y_test, predictions, model_name)

            results[model_name] = {**training_results, **prediction_results, **evaluation_results}
            save_model(model, model_name, scenario_name, artifacts_dir)

        save_results(results, scenario_name, results_dir)


if __name__ == "__main__":
    main()
