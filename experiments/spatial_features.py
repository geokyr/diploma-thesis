from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import add_spatial_features, split_features_and_target
from thesis.eta.models import ModelType, create_model, save_model
from thesis.eta.pipeline import evaluate_predictions, get_stratified_kfold_cv, make_predictions, train_model
from thesis.eta.results import build_cv_results, build_model_results, save_results


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train_raw = generate_trips(fcd_train)
    trips_train = add_spatial_features(trips_train_raw)
    X_train, y_train = split_features_and_target(trips_train)

    skf, stratify_key = get_stratified_kfold_cv(y_train)
    results = {}

    for model_type in ModelType:
        model = create_model(model_type)

        per_fold_results = []

        for train_index, val_index in skf.split(X_train, stratify_key):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            training_results = train_model(model, model_type, X_train_fold, y_train_fold)
            predictions, prediction_results = make_predictions(model, model_type, X_val_fold)
            evaluation_results = evaluate_predictions(y_val_fold, predictions, model_type)

            fold_results = build_model_results(training_results, prediction_results, evaluation_results)
            per_fold_results.append(fold_results)

        results[model_type] = build_cv_results(per_fold_results)

        logger.info(f"Training final {model_type} model on all training data")
        final_model = create_model(model_type)
        train_model(final_model, model_type, X_train, y_train)
        save_model(final_model, model_type, experiment.models_dir)

    save_results(results, experiment.results_dir)


if __name__ == "__main__":
    main()
