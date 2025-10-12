from thesis.common.config import APPDATA_DIRNAME, MISC_DIRNAME, PROJECT_DIR, SMOOTHING_WINDOW_SAMPLES
from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.enums import MLTask
from thesis.common.logger import setup_logger
from thesis.drift.services.detector_manager import DetectorManager
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import add_all_features, split_features_and_target
from thesis.eta.models import ModelType, load_model
from thesis.eta.pipeline import compute_absolute_errors, make_predictions


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train_raw = generate_trips(fcd_train)
    trips_train = add_all_features(trips_train_raw)
    X_train, y_train = split_features_and_target(trips_train)

    model_type = ModelType.LIGHTGBM_REGRESSOR
    model = load_model(model_type, experiment.final_models_dir)

    predictions, _ = make_predictions(model, model_type, X_train)
    absolute_errors = compute_absolute_errors(y_train, predictions)

    misc_dir = PROJECT_DIR / APPDATA_DIRNAME / MISC_DIRNAME
    ml_task = MLTask.ETA
    detector_manager = DetectorManager(misc_dir, ml_task, SMOOTHING_WINDOW_SAMPLES)
    detector_manager.calibrate(absolute_errors)
    detector_manager.save()


if __name__ == "__main__":
    main()
