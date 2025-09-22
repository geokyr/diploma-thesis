"""Data loader for a predictor."""

from pathlib import Path

import pandas as pd

from thesis.common.config import TIME_START_COLUMN, TRIPS_PARQUET_FILENAME


class DataLoader:
    """Data loader for a predictor."""

    def __init__(self, data_dir: Path) -> None:
        self._data_path: Path = data_dir / TRIPS_PARQUET_FILENAME

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
                    (TIME_START_COLUMN, ">=", start_timestamp),
                    (TIME_START_COLUMN, "<", end_timestamp),
                ]
            ],
        )

    def clear(self) -> None:
        """Clear the data loader."""
        pass
