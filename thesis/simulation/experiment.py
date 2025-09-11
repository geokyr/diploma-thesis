"""Simulation experiment configuration and management utilities."""

from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import (
    DATA_DIRNAME,
    GUI_SETTINGS_FILENAME,
    LOGS_DIRNAME,
    NETWORK_BASE_FILENAME,
    NETWORK_RAIN_FILENAME,
    OSM_DATA_FILENAME,
    PLOTS_DIRNAME,
    POLY_FILENAME,
    SIMULATION_DIR,
)


@dataclass(frozen=True, slots=True)
class SimulationExperiment:
    """
    A simulation experiment.

    Attributes:
        simulation_dir (Path): Directory for the simulation experiment.

    Properties:
        name (str): Name of the simulation experiment.
        data_dir (Path): Subdirectory for the data.
        logs_dir (Path): Subdirectory for the logs.
        plots_dir (Path): Subdirectory for the plots.
        osm_data_path (Path): Path to the osm data file.
        gui_settings_path (Path): Path to the gui settings file.
        poly_path (Path): Path to the poly file.
        network_base_path (Path): Path to the network file.
        network_rain_path (Path): Path to the rain network file.
    """

    simulation_dir: Path = SIMULATION_DIR

    def __post_init__(self) -> None:
        for dir in [self.data_dir, self.logs_dir, self.plots_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.simulation_dir=}, "
            f"{self.name=}, "
            f"{self.data_dir=}, "
            f"{self.logs_dir=}, "
            f"{self.plots_dir=}, "
            f"{self.osm_data_path=}, "
            f"{self.gui_settings_path=}, "
            f"{self.poly_path=}, "
            f"{self.network_base_path=}, "
            f"{self.network_rain_path=})"
        )

    @property
    def name(self) -> str:
        return self.simulation_dir.stem

    @property
    def data_dir(self) -> Path:
        return self.simulation_dir / DATA_DIRNAME

    @property
    def logs_dir(self) -> Path:
        return self.simulation_dir / LOGS_DIRNAME

    @property
    def plots_dir(self) -> Path:
        return self.simulation_dir / PLOTS_DIRNAME

    @property
    def osm_data_path(self) -> Path:
        return self.simulation_dir / OSM_DATA_FILENAME

    @property
    def gui_settings_path(self) -> Path:
        return self.simulation_dir / GUI_SETTINGS_FILENAME

    @property
    def poly_path(self) -> Path:
        return self.simulation_dir / POLY_FILENAME

    @property
    def network_base_path(self) -> Path:
        return self.simulation_dir / NETWORK_BASE_FILENAME

    @property
    def network_rain_path(self) -> Path:
        return self.simulation_dir / NETWORK_RAIN_FILENAME
