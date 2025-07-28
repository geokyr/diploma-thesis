from enum import StrEnum


class SimulationScenario(StrEnum):
    """
    Simulation scenarios.

    Attributes:
        TRAIN: Train data on base network.
        TEST: Test data on base network.
        RAIN: Retrain/test data on rain network.
    """

    TRAIN = "train"
    TEST = "test"
    RAIN = "rain"


class ETAScenario(StrEnum):
    """
    ETA scenarios.

    Attributes:
        RESEARCH: Cross-validation model research on train data.
        STABLE: Stable performance evaluation on clean test data.
        DRIFT: Concept drift impact when evaluating on rain data.
        ADAPTATION: Adaptation effectiveness after retraining on rain data.
    """

    RESEARCH = "research"
    STABLE = "stable"
    DRIFT = "drift"
    ADAPTATION = "adaptation"


