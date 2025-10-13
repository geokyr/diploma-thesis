from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import FeatureCalibratorETA, split_features_and_target
from thesis.eta.models import ModelType, create_model, save_model
from thesis.eta.pipeline import get_stratified_kfold_cv, train_model
from thesis.eta.results import build_tuning_results, save_tuning_results
from thesis.eta.tuners import XGBoostTuner, run_hyperparameter_tuning


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

    tuner = XGBoostTuner()
    study = run_hyperparameter_tuning(tuner, trips_train_raw, skf, stratify_key, experiment.name)
    results = build_tuning_results(study)
    save_tuning_results(results, experiment.results_dir)

    model_type = ModelType.XGBOOST_REGRESSOR
    logger.info(f"Training final {model_type} with best parameters on all training data")
    calibrator = FeatureCalibratorETA.from_train_trips(trips_train_raw)
    trips_train = calibrator.transform(trips_train_raw)
    X_train, y_train = split_features_and_target(trips_train)

    model = create_model(model_type, **study.best_params)
    train_model(model, model_type, X_train, y_train)
    save_model(model, model_type, experiment.models_dir)


if __name__ == "__main__":
    main()
