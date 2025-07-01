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

ZENODO_BASE_URL = "https://zenodo.org/records/15402213"
DATASET_FILES_MD5 = {
    "base-test-dump.csv": "d3cf8d0f7da4ea9cd451bede43d982f8",
    "base-test-emission.csv": "7b09ebf3130a7243aa898f19d72fafce",
    "base-test-fcd.csv": "1e37967bf60494082b97e5c3ef37927a",
    "base-train-dump.csv": "07e0bbb2d8fa0905b5544310c5175305",
    "base-train-emission.csv": "11d19255acee5a88204b4abfbbe675a6",
    "base-train-fcd.csv": "4814aeb4174b1de5a39dbb163df9d888",
    "closure-test-dump.csv": "0113c204a794bfb430495dc77b2e0fa3",
    "closure-test-emission.csv": "48174bb3907298dfcb89388a8de05e48",
    "closure-test-fcd.csv": "a7978981b1d41592378477395364b390",
    "closure-train-dump.csv": "59d86fd0fd7f1545e489c4067fdb9b88",
    "closure-train-emission.csv": "0cc63ed61703b4ddada12c0b6465ea2f",
    "closure-train-fcd.csv": "8251a99deba11266d0cec0bf044a38f5",
    "rain-test-dump.csv": "b1df2a914c734816b0b7fcb15f8bea29",
    "rain-test-emission.csv": "ca1e63a9aefd4fceac4fc0f8ce77b3b0",
    "rain-test-fcd.csv": "15ff6d54de5fd06e5ba3c2e6e8bbd95c",
    "rain-train-dump.csv": "959ea1f73dc9bca7937efb7d59c70823",
    "rain-train-emission.csv": "2edb39d330a5cb6247c8636b8f2ca313",
    "rain-train-fcd.csv": "f13590799bdb80c4b05fcc0b62c40c11",
}

SCENARIOS = ["base", "closure", "rain", "base-closure", "base-rain"]
SCENARIOS_SPECS = [
    (
        scenario,
        DATA_DIR / f"{scenario.partition('-')[0]}-train-fcd.csv",
        DATA_DIR / f"{scenario.partition('-')[2] or scenario}-test-fcd.csv",
    )
    for scenario in SCENARIOS
]
