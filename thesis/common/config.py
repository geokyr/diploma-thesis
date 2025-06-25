from pathlib import Path

import numpy as np

np.random.seed(42)

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.45, 0.50, 0.65, 0.75, 0.80, 0.80, 0.75, 0.55, 0.50, 0.55]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.uniform(0.98, 1.02) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]

TYPE_TRAIN = "train"
TYPE_TEST = "test"

PROJECT_ROOT = Path(__file__).parent.parent.parent
SIMULATION_DIR = PROJECT_ROOT / "simulation"
DATA_DIR = SIMULATION_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
