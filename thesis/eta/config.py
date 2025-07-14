from thesis.common.config import DATA_DIR, PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LR = "lr"
XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
CATBOOST = "catboost"

ZENODO_DATASET_API_URL = "https://zenodo.org/api/records/15848647"

SCENARIOS = ["base", "rain", "base-rain"]
SCENARIOS_SPECS = [
    (
        scenario,
        DATA_DIR / f"{scenario.partition('-')[0]}-train-fcd.csv",
        DATA_DIR / f"{scenario.partition('-')[2] or scenario}-test-fcd.csv",
    )
    for scenario in SCENARIOS
]
