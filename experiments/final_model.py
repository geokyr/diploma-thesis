from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment, save_model
from thesis.eta.features import add_all_features, create_quantile_transformer, split_features_and_target
from thesis.eta.models import ModelType, create_model, wrap_with_transformed_target_regressor
from thesis.eta.pipeline import train_model


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
    best_params = {
        "max_depth": 14,
        "n_estimators": 1400,
        "num_leaves": 120,
        "learning_rate": 0.03171569744780489,
        "subsample": 0.632766493612861,
        "colsample_bytree": 0.8591507913974749,
        "min_child_samples": 43,
        "min_split_gain": 0.00098582036201482,
        "reg_lambda": 0.01019913851411128,
        "reg_alpha": 0.0,
    }

    model_type = ModelType.LIGHTGBM_REGRESSOR
    logger.info(f"Training final {model_type} with best parameters on all training data")
    base_model = create_model(model_type, **best_params)
    final_model = wrap_with_transformed_target_regressor(base_model, transformer)
    train_model(final_model, model_type, X_train, y_train)
    save_model(final_model, model_type, experiment.models_dir)


if __name__ == "__main__":
    main()
