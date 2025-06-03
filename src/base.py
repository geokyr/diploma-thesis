import json
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def load_and_prepare(fcd_path):
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


def make_features(df):
    X = df[["origin_x", "origin_y", "dest_x", "dest_y", "hour", "distance"]]
    y = df["duration"]
    return X, y


def train_and_evaluate(X_train, y_train, X_test, y_test, scenario_name):
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

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
        print(f"{name} - MAE: {mae:.2f}s, RMSE: {rmse:.2f}s, MAPE: {mape * 100:.2f}%")

        # Save the model
        model_path = os.path.join("artifacts", "models", f"{scenario_name}-{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    return results


if __name__ == "__main__":
    # Silence a WinError2 about core count
    os.environ["LOKY_MAX_CPU_COUNT"] = "12"

    scenarios = [
        ("base", "data/1.0.0/base-train-fcd.csv", "data/1.0.0/base-test-fcd.csv"),
        ("closure", "data/1.0.0/closure-train-fcd.csv", "data/1.0.0/closure-test-fcd.csv"),
        ("rain", "data/1.0.0/rain-train-fcd.csv", "data/1.0.0/rain-test-fcd.csv"),
    ]

    all_results = {}
    for name, train_path, test_path in scenarios:
        print(f"Scenario: {name}")

        train_trips = load_and_prepare(train_path)
        test_trips = load_and_prepare(test_path)

        X_train, y_train = make_features(train_trips)
        X_test, y_test = make_features(test_trips)

        res = train_and_evaluate(X_train, y_train, X_test, y_test, name)
        all_results[name] = res

    print("\nAll Results:")
    print(json.dumps(all_results, indent=2))

    with open("artifacts/results/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to artifacts/results/results.json")
