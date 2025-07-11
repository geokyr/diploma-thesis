from pathlib import Path

import numpy as np

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TRAIN_TRAFFIC_GENERATION_PERIODS = [0.30, 0.35, 0.45, 0.50, 0.55, 0.55, 0.50, 0.35, 0.35, 0.40]
TEST_TRAFFIC_GENERATION_PERIODS = [p * np.random.normal(1.0, 0.01) for p in TRAIN_TRAFFIC_GENERATION_PERIODS]

TYPE_TRAIN = "train"
TYPE_TEST = "test"

PROJECT_ROOT = Path(__file__).parent.parent.parent
SIMULATION_DIR = PROJECT_ROOT / "new-simulation"
DATA_DIR = SIMULATION_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
