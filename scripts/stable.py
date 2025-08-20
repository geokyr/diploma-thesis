from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment, save_model
from thesis.eta.features import split_features_and_target
from thesis.eta.models import ModelType, create_model
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model
from thesis.eta.results import build_model_results, save_results


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.STABLE)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train = generate_trips(fcd_train)
    X_train, y_train = split_features_and_target(trips_train)

    ensure_dataset_is_valid(experiment.test_path)
    fcd_test_raw = load_fcd_dataset(experiment.test_path)
    fcd_test = preprocess_fcd_dataset(fcd_test_raw)
    trips_test = generate_trips(fcd_test)
    X_test, y_test = split_features_and_target(trips_test)

    results = {}

    for model_type in ModelType:
        model = create_model(model_type)

        training_results = train_model(model, model_type, X_train, y_train)
        predictions, prediction_results = make_predictions(model, model_type, X_test)
        evaluation_results = evaluate_predictions(y_test, predictions, model_type)

        results[model_type] = build_model_results(training_results, prediction_results, evaluation_results)
        save_model(model, model_type, experiment.models_dir)

    save_results(results, experiment.results_dir)


if __name__ == "__main__":
    main()
