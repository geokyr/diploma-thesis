import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from thesis.common.logger import BASELINE_LOGGER_NAME, LOG_FILES_CONFIG, setup_logger

logger = setup_logger(name=BASELINE_LOGGER_NAME, log_file=LOG_FILES_CONFIG[BASELINE_LOGGER_NAME])


def load_and_prepare(fcd_path: str) -> pd.DataFrame:
    """
    Load and prepare the FCD data for training and evaluation.

    Args:
        fcd_path (str): The path to the FCD data file.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared FCD data in trips format.
    """
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
    return pd.DataFrame(trips)


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Make features for the baseline model.

    Args:
        df (pd.DataFrame): A DataFrame containing the prepared FCD data in trips format.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the features and the target variable.
    """
    X = df[["origin_x", "origin_y", "dest_x", "dest_y", "hour", "distance"]]
    y = df["duration"]
    return X, y


def train_and_evaluate(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, scenario_name: str
) -> dict:
    """
    Train and evaluate the baseline model.

    Args:
        X_train (pd.DataFrame): A DataFrame containing the training features.
        y_train (pd.Series): A Series containing the training target variable.
        X_test (pd.DataFrame): A DataFrame containing the test features.
        y_test (pd.Series): A Series containing the test target variable.
        scenario_name (str): The name of the scenario.

    Returns:
        dict: A dictionary containing the results of the training and evaluation.
    """
    models = {
        "linear-regression": LinearRegression(),
        "xgboost": XGBRegressor(random_state=42),
        "lightgbm": LGBMRegressor(random_state=42, verbose=0),
        "catboost": CatBoostRegressor(random_state=42, verbose=0, allow_writing_files=False),
        "multi-layer-perceptron": MLPRegressor(random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")

        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_end = time.perf_counter()
        train_time = train_end - train_start

        eval_start = time.perf_counter()
        preds = model.predict(X_test)
        eval_end = time.perf_counter()
        eval_time = eval_end - eval_start

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Training": train_time, "Evaluation": eval_time}
        logger.info(
            f"{name} - MAE: {mae:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape * 100:.2f}%, Training: {train_time:.3f}s, Evaluation: {eval_time:.3f}s"
        )

        model_path = os.path.join("artifacts", "baseline", f"{scenario_name}-{name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    return results


if __name__ == "__main__":
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    scenarios = [
        (scenario, f"data/1.0.0/{scenario}-train-fcd.csv", f"data/1.0.0/{scenario}-test-fcd.csv")
        for scenario in ["base", "closure", "rain"]
    ]

    all_results = {}
    for name, train_path, test_path in scenarios:
        logger.info(f"Scenario: {name}")

        train_trips = load_and_prepare(train_path)
        test_trips = load_and_prepare(test_path)

        X_train, y_train = make_features(train_trips)
        X_test, y_test = make_features(test_trips)

        res = train_and_evaluate(X_train, y_train, X_test, y_test, name)
        all_results[name] = res

    logger.info("All Results:")
    logger.info(json.dumps(all_results, indent=2))

    with open("artifacts/baseline/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to artifacts/baseline/results.json")
