import os

from thesis.eta.artifacts import (
    construct_model_results_dict,
    save_experiment_results,
    save_model,
    save_scenario_results,
)
from thesis.eta.config import EXTRA_SCENARIOS_SPECS, SCENARIOS_SPECS
from thesis.eta.data import load_fcd_dataset, prepare_baseline_trips
from thesis.eta.evaluation import evaluate_model
from thesis.eta.features import split_features_and_target
from thesis.eta.models import get_baseline_models
from thesis.eta.training import train_model
from thesis.logger import ETA_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=ETA_LOGGER_NAME, log_file=LOG_FILES_CONFIG[ETA_LOGGER_NAME])


def run_scenario(
    scenario_name: str, train_path: str, test_path: str, experiment_name: str
) -> dict[str, dict[str, float]]:
    """
    Run a complete training and evaluation scenario.

    Args:
        scenario_name (str): The name of the scenario.
        train_path (str): Path to the training FCD data.
        test_path (str): Path to the test FCD data.
        experiment_name (str): The name of the experiment.
    Returns:
        dict[str, dict[str, float]]: Results dictionary for the scenario.
    """
    logger.info(f"Starting scenario {scenario_name}...")

    train_df = load_fcd_dataset(train_path)
    test_df = load_fcd_dataset(test_path)

    train_trips = prepare_baseline_trips(train_df)
    test_trips = prepare_baseline_trips(test_df)

    X_train, y_train = split_features_and_target(train_trips)
    X_test, y_test = split_features_and_target(test_trips)

    models = get_baseline_models()

    scenario_results = {}
    for model_name, model in models.items():
        training_time = train_model(model, model_name, X_train, y_train)
        evaluation_time, mae, rmse, mape = evaluate_model(model, model_name, X_test, y_test)

        save_model(model, model_name, scenario_name, experiment_name)
        model_results = construct_model_results_dict(training_time, evaluation_time, mae, rmse, mape)
        scenario_results[model_name] = model_results

    save_scenario_results(scenario_results, scenario_name, experiment_name)
    logger.info(f"Scenario {scenario_name} completed.")
    return scenario_results


def main() -> None:
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    EXPERIMENT_NAME = "baseline"
    logger.info(f"Starting experiment {EXPERIMENT_NAME}...")

    experiment_results = {}
    for scenario_name, train_path, test_path in SCENARIOS_SPECS + EXTRA_SCENARIOS_SPECS:
        scenario_results = run_scenario(scenario_name, train_path, test_path, EXPERIMENT_NAME)
        experiment_results[scenario_name] = scenario_results

    save_experiment_results(experiment_results, EXPERIMENT_NAME)
    logger.info(f"Experiment {EXPERIMENT_NAME} completed.")


if __name__ == "__main__":
    main()
