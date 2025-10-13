from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.common.enums import FeatureGroup
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment
from thesis.eta.features import (
    FeatureCalibrator,
    combine_feature_selection_results,
    get_average_shap_importance,
    get_shap_importance,
    save_feature_selection_results,
    split_features_and_target,
)
from thesis.eta.models import ModelType, create_model
from thesis.eta.pipeline import get_stratified_kfold_cv, train_model


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
    feature_groups = tuple(FeatureGroup)
    model_types = (ModelType.LIGHTGBM_REGRESSOR, ModelType.XGBOOST_REGRESSOR)

    model_shap_importances = {}

    for model_type in model_types:
        logger.info(f"Computing SHAP importance for {model_type}")

        per_fold_shap_importances = []

        for train_index, validation_index in skf.split(trips_train_raw, stratify_key):
            trips_train_fold_raw = trips_train_raw.iloc[train_index].copy()
            trips_validation_fold_raw = trips_train_raw.iloc[validation_index].copy()

            calibrator = FeatureCalibrator.from_train_trips(trips_train_fold_raw, feature_groups=feature_groups)
            trips_train_fold = calibrator.transform(trips_train_fold_raw)
            trips_validation_fold = calibrator.transform(trips_validation_fold_raw)
            X_train_fold, y_train_fold = split_features_and_target(trips_train_fold)
            X_validation_fold, _ = split_features_and_target(trips_validation_fold)

            model = create_model(model_type)
            train_model(model, model_type, X_train_fold, y_train_fold)

            fold_shap_importance = get_shap_importance(model, X_validation_fold)
            per_fold_shap_importances.append(fold_shap_importance)

        average_shap_importance = get_average_shap_importance(per_fold_shap_importances)
        model_shap_importances[model_type] = average_shap_importance

    feature_selection_results = combine_feature_selection_results(model_shap_importances)
    save_feature_selection_results(feature_selection_results, experiment.results_dir)


if __name__ == "__main__":
    main()
