from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

SCENARIOS = ["base", "closure", "rain"]
SCENARIOS_SPECS = [
    (scenario, DATA_DIR / "1.0.0" / f"{scenario}-train-fcd.csv", DATA_DIR / "1.0.0" / f"{scenario}-test-fcd.csv")
    for scenario in SCENARIOS
]

EXTRA_SCENARIOS = ["base_closure", "base_rain"]
EXTRA_SCENARIOS_SPECS = [
    (
        scenario,
        DATA_DIR / "1.0.0" / f"{scenario.split('_')[0]}-train-fcd.csv",
        DATA_DIR / "1.0.0" / f"{scenario.split('_')[1]}-test-fcd.csv",
    )
    for scenario in EXTRA_SCENARIOS
]
