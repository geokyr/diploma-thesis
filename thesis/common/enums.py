from enum import StrEnum


class SimulationScenario(StrEnum):
    TRAIN = "train"
    TEST = "test"
    RAIN = "rain"


# TODO
class Recipe(StrEnum):
    BASE = "base"
    BASE_RAIN = "base-rain"
    RAIN = "rain"


# TODO
class Split(StrEnum):
    TRAIN = "train"
    TEST = "test"


# TODO
class Model(StrEnum):
    LINEAR_REGRESSION = "linear_regression"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
