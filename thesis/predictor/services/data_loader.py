"""Data loader for a predictor."""

from pathlib import Path

import pandas as pd

from thesis.common.config import TIME_START_COLUMN_ETA, TIME_START_COLUMN_FUEL, TRIPS_PARQUET_FILENAME
from thesis.common.enums import MLTask

# TODO: add stops
_TIME_START_COLUMN_MAP = {
    MLTask.ETA: TIME_START_COLUMN_ETA,
    MLTask.FUEL: TIME_START_COLUMN_FUEL,
    MLTask.STOPS: "",
}


class DataLoader:
    """Data loader for a predictor."""

    def __init__(self, data_dir: Path, ml_task: MLTask) -> None:
        self._data_path: Path = data_dir / TRIPS_PARQUET_FILENAME
        self._time_start_column: str = _TIME_START_COLUMN_MAP[ml_task]

    def load_window(self, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        """
        Load a window of data.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            pd.DataFrame: Window of data.
        """
        return pd.read_parquet(
            self._data_path,
            filters=[
                [
                    (self._time_start_column, ">=", start_timestamp),
                    (self._time_start_column, "<", end_timestamp),
                ]
            ],
        )

    def clear(self) -> None:
        """Clear the data loader."""
        pass
