from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.enums import FeatureGroup
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import FeatureCalibrator, split_features_and_target
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

    _, y_split = split_features_and_target(trips_train_raw)
    skf, stratify_key = get_stratified_kfold_cv(y_split)
    feature_groups = (FeatureGroup.FOURIER,)
    results = {}

    for model_type in ModelType:
        per_fold_results = []

        for train_index, validation_index in skf.split(trips_train_raw, stratify_key):
            trips_train_fold_raw = trips_train_raw.iloc[train_index].copy()
            trips_validation_fold_raw = trips_train_raw.iloc[validation_index].copy()

            calibrator = FeatureCalibrator.from_train_trips(trips_train_fold_raw, feature_groups=feature_groups)

            trips_train_fold = calibrator.transform(trips_train_fold_raw)
            trips_validation_fold = calibrator.transform(trips_validation_fold_raw)

            X_train_fold, y_train_fold = split_features_and_target(trips_train_fold)
            X_validation_fold, y_validation_fold = split_features_and_target(trips_validation_fold)

            model = create_model(model_type)
            training_results = train_model(model, model_type, X_train_fold, y_train_fold)
            predictions, prediction_results = make_predictions(model, model_type, X_validation_fold)
            evaluation_results = evaluate_predictions(y_validation_fold, predictions, model_type)

            fold_results = build_model_results(training_results, prediction_results, evaluation_results)
            per_fold_results.append(fold_results)

        results[model_type] = build_cv_results(per_fold_results)

        logger.info(f"Training final {model_type} model on all training data")
        final_calibrator = FeatureCalibrator.from_train_trips(trips_train_raw, feature_groups=feature_groups)
        trips_train = final_calibrator.transform(trips_train_raw)
        X_train, y_train = split_features_and_target(trips_train)

        final_model = create_model(model_type)
        train_model(final_model, model_type, X_train, y_train)
        save_model(final_model, model_type, experiment.models_dir)

    save_results(results, experiment.results_dir)


if __name__ == "__main__":
    main()
