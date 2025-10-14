from thesis.common.data import (
    generate_trips,
    get_trips_parquet_path,
    load_fcd_dataset,
    merge_test_and_rain_trips,
    preprocess_fcd_dataset,
)
from thesis.common.enums import ETAEvaluation, MLTask
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAExperiment
from thesis.eta.features import FeatureCalibratorETA


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train = generate_trips(fcd_train)

    ensure_dataset_is_valid(experiment.test_path)
    fcd_test_raw = load_fcd_dataset(experiment.test_path)
    fcd_test = preprocess_fcd_dataset(fcd_test_raw)
    trips_test = generate_trips(fcd_test)

    ensure_dataset_is_valid(experiment.rain_path)
    fcd_rain_raw = load_fcd_dataset(experiment.rain_path)
    fcd_rain = preprocess_fcd_dataset(fcd_rain_raw)
    trips_rain = generate_trips(fcd_rain)

    calibrator = FeatureCalibratorETA.from_train_trips(trips_train)
    feature_calibrator_dir = calibrator.get_feature_calibrator_dir(MLTask.ETA)
    calibrator.save(feature_calibrator_dir)

    trips_merged_raw = merge_test_and_rain_trips(trips_test, trips_rain)
    trips_merged = calibrator.transform(trips_merged_raw)

    trips_parquet_path = get_trips_parquet_path(MLTask.ETA)
    trips_merged.to_parquet(trips_parquet_path)

    logger.info(f"Precomputed features for {len(trips_merged)} trips: {trips_parquet_path}")


if __name__ == "__main__":
    main()
