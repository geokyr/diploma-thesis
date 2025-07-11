from thesis.common.logger import setup_logger
from thesis.eta.config import SCENARIOS_SPECS
from thesis.eta.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.eta.experiment import initialize_experiment, save_model, save_results
from thesis.eta.features import (
    engineer_trip_features,
    fit_and_transform_scaling,
    log_transform,
    reverse_log_transform,
    split_features_and_target,
)
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model


def main() -> None:
    experiment_name = "feature_engineering"
    artifacts_dir, logs_dir, _, results_dir = initialize_experiment(experiment_name)
    logger = setup_logger(experiment_name, logs_dir)

    for scenario_name, train_path, test_path in SCENARIOS_SPECS:
        logger.info(f"Starting scenario {scenario_name}")

        # Load and prepare data
        fcd_train = load_fcd_dataset(train_path)
        fcd_test = load_fcd_dataset(test_path)
        fcd_train = preprocess_fcd_dataset(fcd_train)
        fcd_test = preprocess_fcd_dataset(fcd_test)
        trips_train = generate_trips(fcd_train)
        trips_test = generate_trips(fcd_test)

        # Engineer features BEFORE log transformation
        logger.info("Engineering features")
        trips_train = engineer_trip_features(trips_train)
        trips_test = engineer_trip_features(trips_test)

        # Apply log transformation to distance and duration
        logger.info("Applying log transformation to distance and duration")
        trips_train = log_transform(trips_train)
        trips_test = log_transform(trips_test)

        # Split features and target
        X_train, y_train = split_features_and_target(trips_train)
        X_test, y_test = split_features_and_target(trips_test)

        # Apply scaling to features (excluding categorical-like features)
        logger.info("Applying standard scaling to features")
        exclude_from_scaling = ["hour_bin", "is_morning_rush", "is_evening_rush", "is_night"]
        X_train, X_test = fit_and_transform_scaling(X_train, X_test, exclude_columns=exclude_from_scaling)

        # Log feature information
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Features: {list(X_train.columns)}")

        # Train and evaluate models
        models = get_baseline_models()
        results = {}
        for model_name, model in models.items():
            logger.info(f"Training and evaluating {model_name}")

            # Train model (in log space for target)
            training_results = train_model(model, model_name, X_train, y_train)

            # Make predictions (in log space)
            predictions_log, prediction_results = make_predictions(model, model_name, X_test)

            # Transform predictions and true values back to original scale for evaluation
            predictions_original = reverse_log_transform(predictions_log)
            y_test_original = reverse_log_transform(y_test)

            # Evaluate on original scale
            evaluation_results = evaluate_predictions(y_test_original, predictions_original, model_name)

            # Combine all results
            model_results = {**training_results, **prediction_results, **evaluation_results}
            results[model_name] = model_results

            save_model(model, model_name, scenario_name, artifacts_dir)

        save_results(results, scenario_name, results_dir)


if __name__ == "__main__":
    main()
