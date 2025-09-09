from thesis.common.config import OUTPUTS_DIR
from thesis.common.data import (
    generate_trips,
    get_adaptation_retrain_data,
    get_adaptation_test_data,
    load_fcd_dataset,
    preprocess_fcd_dataset,
)
from thesis.common.logger import setup_logger
from thesis.eta.data import ensure_dataset_is_valid
from thesis.eta.experiment import ETAEvaluation, ETAExperiment, load_model
from thesis.eta.features import add_all_features, split_features_and_target
from thesis.eta.models import ModelType, create_model, get_retraining_kwargs
from thesis.eta.pipeline import evaluate_predictions, make_predictions, train_model


def main() -> None:
    experiment = ETAExperiment(__file__, ETAEvaluation.RESEARCH)
    logger = setup_logger(experiment.name, experiment.logs_dir)

    logger.info(f"Running {experiment.name} experiment with {experiment.evaluation} evaluation")

    lightgbm_models_dir = OUTPUTS_DIR / "lightgbm_tuning_focused" / "models"
    catboost_models_dir = OUTPUTS_DIR / "catboost_tuning_focused" / "models"
    xgboost_models_dir = OUTPUTS_DIR / "xgboost_tuning_focused" / "models"

    logger.info(f"{ETAEvaluation.STABLE} - Evaluating on test data")

    ensure_dataset_is_valid(experiment.test_path)
    fcd_test_raw = load_fcd_dataset(experiment.test_path)
    fcd_test = preprocess_fcd_dataset(fcd_test_raw)
    trips_test_raw = generate_trips(fcd_test)
    trips_test = add_all_features(trips_test_raw)
    X_test, y_test = split_features_and_target(trips_test)

    model_type = ModelType.LIGHTGBM_REGRESSOR
    model = load_model(model_type, lightgbm_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.CATBOOST_REGRESSOR
    model = load_model(model_type, catboost_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.XGBOOST_REGRESSOR
    model = load_model(model_type, xgboost_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    logger.info(f"{ETAEvaluation.DRIFT} - Evaluating on drift rain data")

    ensure_dataset_is_valid(experiment.rain_path)
    fcd_rain_raw = load_fcd_dataset(experiment.rain_path)
    fcd_rain = preprocess_fcd_dataset(fcd_rain_raw)
    trips_rain_raw = generate_trips(fcd_rain)
    trips_rain = add_all_features(trips_rain_raw)
    test_data = get_adaptation_test_data(trips_rain)
    X_test, y_test = split_features_and_target(test_data)

    model_type = ModelType.LIGHTGBM_REGRESSOR
    model = load_model(model_type, lightgbm_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.CATBOOST_REGRESSOR
    model = load_model(model_type, catboost_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.XGBOOST_REGRESSOR
    model = load_model(model_type, xgboost_models_dir)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    logger.info(f"{ETAEvaluation.RETRAIN} - Evaluating retrained model")

    retrain_data = get_adaptation_retrain_data(trips_test, trips_rain)
    test_data = get_adaptation_test_data(trips_rain)
    X_retrain, y_retrain = split_features_and_target(retrain_data)
    X_test, y_test = split_features_and_target(test_data)

    lightgbm_kwargs = {
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

    catboost_kwargs = {
        "n_estimators": 1400,
        "max_depth": 10,
        "learning_rate": 0.025347226438749645,
        "subsample": 0.7914420270077684,
        "colsample_bylevel": 0.8780092655440341,
        "l2_leaf_reg": 9.686385972818426,
        "random_strength": 0.12109306207231453,
        "bagging_temperature": 1.7070606903680519,
        "min_data_in_leaf": 37,
        "border_count": 255,
    }

    xgboost_kwargs = {
        "n_estimators": 1400,
        "max_depth": 8,
        "learning_rate": 0.029556766569162514,
        "subsample": 0.5631288685406436,
        "colsample_bytree": 0.6806856533241848,
        "colsample_bynode": 0.8624780133289642,
        "min_child_weight": 14,
        "gamma": 0.08144185211536754,
        "reg_alpha": 0.6707842755090976,
        "reg_lambda": 6.477228902914716e-05,
    }

    model_type = ModelType.LIGHTGBM_REGRESSOR
    model = create_model(model_type, **lightgbm_kwargs)
    trained_model = load_model(model_type, lightgbm_models_dir)
    retraining_kwargs = get_retraining_kwargs(model_type, trained_model)
    _ = train_model(model, model_type, X_retrain, y_retrain, **retraining_kwargs)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.CATBOOST_REGRESSOR
    model = create_model(model_type, **catboost_kwargs)
    trained_model = load_model(model_type, catboost_models_dir)
    retraining_kwargs = get_retraining_kwargs(model_type, trained_model)
    _ = train_model(model, model_type, X_retrain, y_retrain, **retraining_kwargs)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)

    model_type = ModelType.XGBOOST_REGRESSOR
    model = create_model(model_type, **xgboost_kwargs)
    trained_model = load_model(model_type, xgboost_models_dir)
    retraining_kwargs = get_retraining_kwargs(model_type, trained_model)
    _ = train_model(model, model_type, X_retrain, y_retrain, **retraining_kwargs)
    predictions, _ = make_predictions(model, model_type, X_test)
    _ = evaluate_predictions(y_test, predictions, model_type)


if __name__ == "__main__":
    main()
