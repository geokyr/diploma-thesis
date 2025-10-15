"""ETA experiments and experiment configuration management."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from thesis.common.config import (
    DATA_DIRNAME,
    FCD_PARQUET_SUFFIX,
    LOGS_DIRNAME,
    MODELS_DIRNAME,
    OUTPUTS_DIR,
    RESULTS_DIRNAME,
    SIMULATION_DIR,
    STABLE_MODELS_DIRNAME,
)
from thesis.common.enums import ETAEvaluation, SimulationScenario

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ETAExperiment:
    """
    An ETA experiment.

    Attributes:
        experiment_filename (str): Name of the file that contains the ETA experiment.
        evaluation (ETAEvaluation): Evaluation of the experiment.
    """

    experiment_filename: str
    evaluation: ETAEvaluation

    _DATA_DIR: ClassVar[Path] = SIMULATION_DIR / DATA_DIRNAME
    _TRAIN_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.TRAIN}{FCD_PARQUET_SUFFIX}"
    _TEST_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.TEST}{FCD_PARQUET_SUFFIX}"
    _RAIN_PATH: ClassVar[Path] = _DATA_DIR / f"{SimulationScenario.RAIN}{FCD_PARQUET_SUFFIX}"

    _STABLE_MODELS_DIR: ClassVar[Path] = OUTPUTS_DIR / STABLE_MODELS_DIRNAME / MODELS_DIRNAME

    def __post_init__(self) -> None:
        for directory in [ETAExperiment._DATA_DIR, self.models_dir, self.logs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.experiment_filename=}, "
            f"{self.evaluation=}, "
            f"{self.name=}, "
            f"{self.experiment_dir=}, "
            f"{self.models_dir=}, "
            f"{self.logs_dir=}, "
            f"{self.results_dir=}, "
            f"{self.train_path=}, "
            f"{self.test_path=}, "
            f"{self.rain_path=}, "
            f"{self.stable_models_dir=})"
        )

    @property
    def name(self) -> str:
        """Name of the ETA experiment."""
        return Path(self.experiment_filename).stem

    @property
    def experiment_dir(self) -> Path:
        """Directory for the ETA experiment."""
        return OUTPUTS_DIR / self.name

    @property
    def models_dir(self) -> Path:
        """Subdirectory for the models."""
        return self.experiment_dir / MODELS_DIRNAME

    @property
    def logs_dir(self) -> Path:
        """Subdirectory for the logs."""
        return self.experiment_dir / LOGS_DIRNAME

    @property
    def results_dir(self) -> Path:
        """Subdirectory for the results."""
        return self.experiment_dir / RESULTS_DIRNAME

    @property
    def train_path(self) -> Path:
        """Path to the train fcd parquet file."""
        return ETAExperiment._TRAIN_PATH

    @property
    def test_path(self) -> Path:
        """Path to the test fcd parquet file."""
        return ETAExperiment._TEST_PATH

    @property
    def rain_path(self) -> Path:
        """Path to the rain fcd parquet file."""
        return ETAExperiment._RAIN_PATH

    @property
    def stable_models_dir(self) -> Path:
        """Directory for the stable models."""
        return ETAExperiment._STABLE_MODELS_DIR
