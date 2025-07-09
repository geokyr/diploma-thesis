import os
import subprocess

from thesis.common.config import DATA_DIR, PROJECT_ROOT

# Silence a WinError2 about core count
os.environ["LOKY_MAX_CPU_COUNT"] = "10"

USE_GPU = bool(subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode().strip())
GPU_DEVICE = 0 if USE_GPU else None
GPU_PLATFORM = 0 if USE_GPU else None

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

LR = "lr"
XGBOOST = "xgboost"
LIGHTGBM = "lightgbm"
CATBOOST = "catboost"

ZENODO_DATASET_API_URL = "https://zenodo.org/api/records/15848647"

SCENARIOS = ["base", "base-rain", "rain"]
SCENARIOS_SPECS = [
    (
        scenario,
        DATA_DIR / f"{scenario.partition('-')[0]}-train-fcd.csv",
        DATA_DIR / f"{scenario.partition('-')[2] or scenario}-test-fcd.csv",
    )
    for scenario in SCENARIOS
]
