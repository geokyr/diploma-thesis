from pathlib import Path

import pandas as pd

from thesis.common.config import TRIPS_PARQUET_FILENAME
from thesis.common.service import PlatformServiceConfig


class TripsDataLoader:
    def __init__(self, service_config: PlatformServiceConfig | None = None) -> None:
        self._cfg = service_config or PlatformServiceConfig()

    def _resolve_path(self) -> Path:
        return self._cfg.data_dir / TRIPS_PARQUET_FILENAME

    def load_window(self, start_ts: float, end_ts: float) -> pd.DataFrame:
        path = self._resolve_path()
        return pd.read_parquet(
            path,
            filters=[[("time_start", ">=", start_ts), ("time_start", "<", end_ts)]],
        )
