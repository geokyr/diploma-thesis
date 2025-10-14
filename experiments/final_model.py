from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.enums import MLTask
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import FeatureCalibratorETA, split_features_and_target
from thesis.eta.models import ModelType, create_model, save_final_model, save_model
from thesis.eta.pipeline import train_model


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train_raw = generate_trips(fcd_train)

    calibrator = FeatureCalibratorETA.from_train_trips(trips_train_raw)
    trips_train = calibrator.transform(trips_train_raw)
    X_train, y_train = split_features_and_target(trips_train)

    best_params = {
        "max_depth": 14,
        "n_estimators": 1100,
        "num_leaves": 104,
        "learning_rate": 0.043311390827708816,
        "subsample": 0.9584344495013392,
        "colsample_bytree": 0.6921069310836445,
        "min_child_samples": 38,
        "min_split_gain": 1.2499234477818435e-07,
        "reg_alpha": 6.790340597036079e-06,
        "reg_lambda": 3.467795781008837e-05,
    }

    model_type = ModelType.LIGHTGBM_REGRESSOR
    logger.info(f"Training final {model_type} with best parameters on all training data")
    model = create_model(model_type, **best_params)
    train_model(model, model_type, X_train, y_train)
    save_model(model, model_type, experiment.models_dir)
    save_final_model(model, model_type, MLTask.ETA)


if __name__ == "__main__":
    main()
