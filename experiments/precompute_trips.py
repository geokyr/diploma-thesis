import pandas as pd

from thesis.common.config import (
    APPDATA_DIRNAME,
    DATA_DIRNAME,
    END_TIME,
    MERGED_TRIPS_FILENAME,
    MISC_DIRNAME,
    PROJECT_DIR,
)
from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.common.service import MLTask
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import FeatureCalibrator


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

    calibrator = FeatureCalibrator.from_train_trips(trips_train)
    misc_dir = PROJECT_DIR / APPDATA_DIRNAME / MISC_DIRNAME / MLTask.ETA
    calibrator.save(misc_dir)

    trips_rain_shifted = trips_rain.assign(time_start=trips_rain["time_start"] + END_TIME)
    trips_merged_raw = (
        pd.concat([trips_test, trips_rain_shifted], ignore_index=True).sort_values("time_start").reset_index(drop=True)
    )
    trips_merged = calibrator.transform(trips_merged_raw)

    out_path = PROJECT_DIR / APPDATA_DIRNAME / DATA_DIRNAME / MLTask.ETA / MERGED_TRIPS_FILENAME
    trips_merged.to_parquet(out_path)

    logger.info(f"Precomputed features for {len(trips_merged)} trips: {out_path}")


if __name__ == "__main__":
    main()
