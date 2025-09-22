"""Parquet data loader for a predictor."""

from pathlib import Path

import pandas as pd

from thesis.common.config import TIME_START_COLUMN, TRIPS_PARQUET_FILENAME


class ParquetDataLoader:
    """Parquet data loader for a predictor."""

    def __init__(self, data_dir: Path) -> None:
        self._data_parquet_path: Path | None = data_dir / TRIPS_PARQUET_FILENAME

    def load_window(self, start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        """
        Load a window of parquet data.

        Args:
            start_timestamp (int): Start timestamp.
            end_timestamp (int): End timestamp.

        Returns:
            pd.DataFrame: Window of parquet data.
        """
        return pd.read_parquet(
            self._data_parquet_path,
            filters=[
                [
                    (TIME_START_COLUMN, ">=", start_timestamp),
                    (TIME_START_COLUMN, "<", end_timestamp),
                ]
            ],
        )

    def close(self) -> None:
        """
        Close the parquet data loader.
        """
        self._data_parquet_path = None
