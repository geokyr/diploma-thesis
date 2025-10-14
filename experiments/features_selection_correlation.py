from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.enums import ETAEvaluation, FeatureGroup
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAExperiment
from thesis.eta.features import (
    FeatureCalibrator,
    compare_correlated_features_pairs,
    find_correlated_feature_pairs,
    load_feature_ranking_results,
    save_feature_selection_results,
    split_features_and_target,
)


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    ensure_dataset_is_valid(experiment.train_path)
    fcd_train_raw = load_fcd_dataset(experiment.train_path)
    fcd_train = preprocess_fcd_dataset(fcd_train_raw)
    trips_train_raw = generate_trips(fcd_train)

    feature_groups = tuple(FeatureGroup)
    calibrator = FeatureCalibrator.from_train_trips(trips_train_raw, feature_groups=feature_groups)
    trips_train = calibrator.transform(trips_train_raw)

    X_train, _ = split_features_and_target(trips_train)

    correlated_feature_pairs = find_correlated_feature_pairs(X_train)
    feature_ranking = load_feature_ranking_results()
    correlated_feature_pairs = compare_correlated_features_pairs(correlated_feature_pairs, feature_ranking)
    save_feature_selection_results(correlated_feature_pairs, experiment.results_dir)


if __name__ == "__main__":
    main()
