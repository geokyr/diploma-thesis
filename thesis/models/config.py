from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BASELINE_ARTIFACTS_DIR = ARTIFACTS_DIR / "baseline"
BASELINE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"

SCENARIOS = ["base", "closure", "rain"]
TRAIN_TEST_SPECS = [
    (scenario, DATA_DIR / "1.0.0" / f"{scenario}-train-fcd.csv", DATA_DIR / "1.0.0" / f"{scenario}-test-fcd.csv")
    for scenario in SCENARIOS
]
