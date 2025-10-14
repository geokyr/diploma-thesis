from thesis.common.data import (
    generate_trips,
    get_adaptation_retrain_data,
    get_adaptation_test_data,
    load_fcd_dataset,
    preprocess_fcd_dataset,
)
from thesis.common.enums import ETAEvaluation, ModelType
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAExperiment
from thesis.eta.features import split_features_and_target
from thesis.eta.models import create_model, get_retraining_kwargs, load_model, save_model
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model
from thesis.eta.results import build_model_results, save_results


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RETRAIN)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} scenario")

    ensure_dataset_is_valid(experiment.test_path)
    fcd_test_raw = load_fcd_dataset(experiment.test_path)
    fcd_test = preprocess_fcd_dataset(fcd_test_raw)
    trips_test = generate_trips(fcd_test)

    ensure_dataset_is_valid(experiment.rain_path)
    fcd_rain_raw = load_fcd_dataset(experiment.rain_path)
    fcd_rain = preprocess_fcd_dataset(fcd_rain_raw)
    trips_rain = generate_trips(fcd_rain)

    retrain_data = get_adaptation_retrain_data(trips_test, trips_rain)
    test_data = get_adaptation_test_data(trips_rain)
    X_retrain, y_retrain = split_features_and_target(retrain_data)
    X_test, y_test = split_features_and_target(test_data)

    results = {}

    for model_type in ModelType:
        model = create_model(model_type)
        trained_model = load_model(model_type, experiment.stable_models_dir)
        retraining_kwargs = get_retraining_kwargs(model_type, trained_model)

        training_results = train_model(model, model_type, X_retrain, y_retrain, **retraining_kwargs)
        predictions, prediction_results = make_predictions(model, model_type, X_test)
        evaluation_results = evaluate_predictions(y_test, predictions, model_type)

        results[model_type] = build_model_results(training_results, prediction_results, evaluation_results)
        save_model(model, model_type, experiment.models_dir)

    save_results(results, experiment.results_dir)


if __name__ == "__main__":
    main()
