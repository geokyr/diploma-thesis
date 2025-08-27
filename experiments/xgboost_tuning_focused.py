from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment, save_model
from thesis.eta.features import add_all_features, create_quantile_transformer, split_features_and_target
from thesis.eta.models import ModelType, create_model, wrap_with_transformed_target_regressor
from thesis.eta.pipeline import get_stratified_kfold_cv, train_model
from thesis.eta.results import build_tuning_results, save_tuning_results
from thesis.eta.tuning import XGBoostTunerFocused, run_hyperparameter_tuning


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train = generate_trips(fcd_train)
    trips_train = add_all_features(trips_train)
    X_train, y_train = split_features_and_target(trips_train)

    transformer = create_quantile_transformer()
    skf, stratify_key = get_stratified_kfold_cv(y_train)

    tuner = XGBoostTunerFocused()
    study = run_hyperparameter_tuning(tuner, X_train, y_train, skf, stratify_key, experiment.name, transformer)
    results = build_tuning_results(study)
    save_tuning_results(results, experiment.results_dir)

    model_type = ModelType.XGBOOST_REGRESSOR
    logger.info(f"Training final {model_type} with best parameters on all training data")
    base_model = create_model(model_type, **study.best_params)
    final_model = wrap_with_transformed_target_regressor(base_model, transformer)
    train_model(final_model, model_type, X_train, y_train)
    save_model(final_model, model_type, experiment.models_dir)


if __name__ == "__main__":
    main()
