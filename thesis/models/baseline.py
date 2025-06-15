import json
import os
import time

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from thesis.common.logger import BASELINE_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger
from thesis.models.config import BASELINE_ARTIFACTS_DIR, EXTRA_SCENARIOS_SPECS, RANDOM_STATE, SCENARIOS_SPECS

logger = setup_logger(name=BASELINE_LOGGER_NAME, log_file=LOG_FILES_CONFIG[BASELINE_LOGGER_NAME])


def load_and_prepare_trips(fcd_path: str) -> pd.DataFrame:
    """
    Load and prepare the trips from FCD data for training and evaluation.

    Args:
        fcd_path (str): The path to the FCD data file.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared FCD data in trips format.
    """
    logger.info(f"Loading FCD data from {fcd_path}...")
    dtype = {
        "timestep_time": int,
        "vehicle_acceleration": float,
        "vehicle_id": str,
        "vehicle_odometer": float,
        "vehicle_speed": float,
        "vehicle_x": float,
        "vehicle_y": float,
    }
    df = pd.read_csv(fcd_path, sep=";", header=0, dtype=dtype)
    logger.info(f"Loaded {len(df)} rows of FCD data.")

    logger.info("Preparing trips...")
    trips = []
    for vehicle_id, group in df.groupby("vehicle_id"):
        start = group.iloc[0]
        end = group.iloc[-1]
        if end["timestep_time"] - start["timestep_time"] <= 0:
            continue
        trips.append(
            {
                "vehicle_id": vehicle_id,
                "origin_x": start["vehicle_x"],
                "origin_y": start["vehicle_y"],
                "dest_x": end["vehicle_x"],
                "dest_y": end["vehicle_y"],
                "hour": start["timestep_time"] // 3600,
                "distance": end["vehicle_odometer"] - start["vehicle_odometer"],
                "duration": end["timestep_time"] - start["timestep_time"],
            }
        )
    trips_df = pd.DataFrame(trips)
    logger.info(f"Prepared {len(trips_df)} trips.")
    return trips_df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Make features for the baseline model.

    Args:
        df (pd.DataFrame): A DataFrame containing the prepared FCD data in trips format.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features and the target variable.
    """
    logger.info("Making features...")
    X = df[["origin_x", "origin_y", "dest_x", "dest_y", "hour", "distance"]]
    y = df["duration"]
    logger.info(f"Made {len(X)} features and {len(y)} target variables.")
    return X, y


def initialize_models() -> dict:
    """
    Initialize the baseline models.

    Args:
        random_state (int): The random state to use for the models.

    Returns:
        dict: A dictionary containing the initialized models.
    """
    logger.info("Initializing models...")
    models = {
        "linear-regression": LinearRegression(),
        "multi-layer-perceptron": MLPRegressor(random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(random_state=RANDOM_STATE),
        "lightgbm": LGBMRegressor(random_state=RANDOM_STATE, verbose=0),
        "catboost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0, allow_writing_files=False),
    }
    logger.info(f"Initialized {len(models)} models.")
    return models


def save_model(model, model_name: str, scenario_name: str) -> None:
    """
    Save a model to the artifacts directory.

    Args:
        model: The machine learning model to save.
        model_name (str): The name of the model.
        scenario_name (str): The name of the scenario.
    """
    model_path = BASELINE_ARTIFACTS_DIR / f"{scenario_name}-{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")


def train_and_evaluate_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scenario_name: str,
) -> dict:
    """
    Train and evaluate a single baseline model.

    Args:
        model: The machine learning model to train and evaluate.
        model_name (str): The name of the model.
        X_train (pd.DataFrame): A DataFrame containing the training features.
        y_train (pd.Series): A Series containing the training target variable.
        X_test (pd.DataFrame): A DataFrame containing the test features.
        y_test (pd.Series): A Series containing the test target variable.
        scenario_name (str): The name of the scenario.

    Returns:
        dict: A dictionary containing the results of the training and evaluation.
    """
    logger.info(f"Training {model_name}...")

    train_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_end = time.perf_counter()
    train_time = train_end - train_start

    eval_start = time.perf_counter()
    preds = model.predict(X_test)
    eval_end = time.perf_counter()
    eval_time = eval_end - eval_start

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)

    logger.info(
        f"{model_name} - MAE: {mae:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape * 100:.2f}%, Training: {train_time:.3f}s, Evaluation: {eval_time:.3f}s"
    )

    save_model(model, model_name, scenario_name)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Training": train_time,
        "Evaluation": eval_time,
    }

    return results


def run_scenario(scenario_name: str, train_path: str, test_path: str) -> dict:
    """
    Run a complete training and evaluation scenario.

    Args:
        scenario_name (str): The name of the scenario.
        train_path (str): Path to the training FCD data.
        test_path (str): Path to the test FCD data.

    Returns:
        dict: Results dictionary for the scenario.
    """
    logger.info(f"Running scenario: {scenario_name}")

    train_trips = load_and_prepare_trips(train_path)
    test_trips = load_and_prepare_trips(test_path)

    X_train, y_train = make_features(train_trips)
    X_test, y_test = make_features(test_trips)

    models = initialize_models()
    scenario_results = {}
    for model_name, model in models.items():
        model_results = train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, scenario_name)
        scenario_results[model_name] = model_results

    logger.info(f"Completed scenario: {scenario_name}")
    return scenario_results


def save_results(results: dict) -> None:
    """
    Save results to a JSON file.

    Args:
        results (dict): The results dictionary to save.
    """
    results_filepath = BASELINE_ARTIFACTS_DIR / "results.json"
    with open(results_filepath, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_filepath}")


def main() -> None:
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    logger.info("Starting baseline evaluation...")
    results = {}

    for scenario_name, train_path, test_path in SCENARIOS_SPECS + EXTRA_SCENARIOS_SPECS:
        scenario_results = run_scenario(scenario_name, train_path, test_path)
        results[scenario_name] = scenario_results

    save_results(results)
    logger.info("Baseline evaluation completed.")


if __name__ == "__main__":
    main()
