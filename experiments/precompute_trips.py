from thesis.common.config import (
    FCD_PARQUET_SUFFIX,
    TRIPS_PARQUET_SUFFIX,
)
from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import add_all_features


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    fcd_paths = [experiment.train_path, experiment.test_path, experiment.rain_path]

    for fcd_path in fcd_paths:
        scenario = fcd_path.name.replace(FCD_PARQUET_SUFFIX, "")
        output_path = fcd_path.with_name(f"{scenario}{TRIPS_PARQUET_SUFFIX}")

        if output_path.exists():
            logger.info(f"Skipping precomputing trips for {scenario} because output already exists: {output_path}")
            continue

        logger.info(f"Precomputing trips for {scenario}")

        ensure_dataset_is_valid(fcd_path)
        fcd_raw = load_fcd_dataset(fcd_path)
        fcd = preprocess_fcd_dataset(fcd_raw)
        trips_raw = generate_trips(fcd)
        trips = add_all_features(trips_raw)
        trips.to_parquet(output_path)

        logger.info(f"Precomputed {len(trips)} trips: {output_path}")


if __name__ == "__main__":
    main()
