from thesis.common.config import PROJECT_ROOT

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LINEAR_REGRESSION = "linear_regression"
XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
CATBOOST = "catboost"

ZENODO_DATASET_API_URL = "https://zenodo.org/api/records/15848647"

SCENARIO_BASE = "base"
SCENARIO_RAIN = "rain"
SCENARIO_BASE_RAIN = "base-rain"
SCENARIOS = [SCENARIO_BASE, SCENARIO_RAIN, SCENARIO_BASE_RAIN]
