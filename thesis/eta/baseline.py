import os

from thesis.eta.artifacts import (
    save_experiment_results,
    save_model,
    save_scenario_results,
)
from thesis.eta.config import ALL_SCENARIOS_SPECS
from thesis.eta.data import clean_fcd_dataset, load_fcd_dataset, prepare_baseline_trips
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_baseline_models
from thesis.eta.pipeline import train_and_evaluate_model
from thesis.logger import setup_logging


def main() -> None:
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    EXPERIMENT_NAME = "baseline"
    logger = setup_logging(EXPERIMENT_NAME)

    logger.info(f"Starting experiment {EXPERIMENT_NAME}")

    experiment_results = {}
    for scenario_name, train_path, test_path in ALL_SCENARIOS_SPECS:
        logger.info(f"Starting scenario {scenario_name}")

        train_df = load_fcd_dataset(train_path)
        test_df = load_fcd_dataset(test_path)
        train_df = clean_fcd_dataset(train_df)
        test_df = clean_fcd_dataset(test_df)
        train_trips = prepare_baseline_trips(train_df)
        test_trips = prepare_baseline_trips(test_df)
        X_train, y_train = split_features_and_target(train_trips)
        X_test, y_test = split_features_and_target(test_trips)

        models = get_baseline_models()
        scenario_results = {}
        for model_name, model in models.items():
            model_results = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test)
            scenario_results[model_name] = model_results
            save_model(model, model_name, scenario_name, EXPERIMENT_NAME)

        experiment_results[scenario_name] = scenario_results
        save_scenario_results(scenario_results, scenario_name, EXPERIMENT_NAME)
        logger.info(f"Completed scenario {scenario_name}")

    save_experiment_results(experiment_results, EXPERIMENT_NAME)
    logger.info(f"Completed experiment {EXPERIMENT_NAME}")


if __name__ == "__main__":
    main()
