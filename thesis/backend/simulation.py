"""
Simulation clock and trip feeder for the platform.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from thesis.backend.models import TripData
from thesis.backend.state import SimulationState
from thesis.common.data import generate_trips, load_fcd_dataset, preprocess_fcd_dataset
from thesis.eta.features import add_all_features

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Manages simulation timing and trip data feeding."""

    def __init__(self, platform_data_path: Path, platform_state_path: Path):
        self.platform_data_path = platform_data_path
        self.platform_state_path = platform_state_path

        self.state = SimulationState.load_checkpoint(platform_state_path)

        # Data storage
        self._test_trips: Optional[pd.DataFrame] = None
        self._rain_trips: Optional[pd.DataFrame] = None
        self._current_trips: Optional[pd.DataFrame] = None

        # Simulation timing
        self.tick_interval_ms: int = 150  # 150ms real time = 1min sim time (400x speedup)
        self.simulation_task: Optional[asyncio.Task] = None

        # Callbacks for event handling
        self.on_metrics_update = None
        self.on_notification = None

        logger.info(f"Simulation engine initialized with state: {self.state}")

    async def initialize_data(self) -> None:
        """Load and preprocess trip data from parquet files."""
        logger.info("Loading simulation data...")

        try:
            # Load test data
            test_path = self.platform_data_path / "test-fcd.parquet"
            if not test_path.exists():
                # Fallback to simulation data directory
                test_path = Path("simulation/data/test-fcd.parquet")

            logger.info(f"Loading test data from {test_path}")
            test_fcd_raw = load_fcd_dataset(test_path)
            test_fcd = preprocess_fcd_dataset(test_fcd_raw)
            self._test_trips = generate_trips(test_fcd)
            self._test_trips = add_all_features(self._test_trips)
            logger.info(f"Loaded {len(self._test_trips)} test trips")

            # Load rain data
            rain_path = self.platform_data_path / "rain-fcd.parquet"
            if not rain_path.exists():
                rain_path = Path("simulation/data/rain-fcd.parquet")

            logger.info(f"Loading rain data from {rain_path}")
            rain_fcd_raw = load_fcd_dataset(rain_path)
            rain_fcd = preprocess_fcd_dataset(rain_fcd_raw)
            self._rain_trips = generate_trips(rain_fcd)
            self._rain_trips = add_all_features(self._rain_trips)
            logger.info(f"Loaded {len(self._rain_trips)} rain trips")

            # Set initial dataset
            self._set_current_dataset(self.state.dataset)

        except Exception as e:
            logger.error(f"Failed to load simulation data: {e}")
            raise

    def _set_current_dataset(self, dataset: str) -> None:
        """Switch between test and rain datasets."""
        if dataset == "test":
            self._current_trips = self._test_trips
        elif dataset == "rain":
            self._current_trips = self._rain_trips
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.state.dataset = dataset
        logger.info(f"Switched to {dataset} dataset with {len(self._current_trips)} trips")

    def get_trip_batch(self, start_time: int, duration_minutes: int) -> Tuple[List[TripData], List[float]]:
        """
        Get batch of trips and ground truth for a time window.

        Returns:
            Tuple of (trip_data_list, ground_truth_durations)
        """
        if self._current_trips is None:
            return [], []

        end_time = start_time + (duration_minutes * 60)

        # Filter trips that start in this time window
        batch_trips = self._current_trips[
            (self._current_trips["time_start"] >= start_time) & (self._current_trips["time_start"] < end_time)
        ].copy()

        if batch_trips.empty:
            return [], []

        # Convert to TripData objects (without ground truth duration)
        trip_data_list = []
        ground_truth = []

        for _, row in batch_trips.iterrows():
            trip_data = TripData(
                source_x=row["source_x"],
                source_y=row["source_y"],
                destination_x=row["destination_x"],
                destination_y=row["destination_y"],
                time_start=row["time_start"],
                distance=row["distance"],
            )
            trip_data_list.append(trip_data)
            ground_truth.append(row["duration"])

        logger.debug(f"Retrieved {len(trip_data_list)} trips for time window {start_time}-{end_time}")
        return trip_data_list, ground_truth

    async def start_simulation(self) -> None:
        """Start the simulation loop."""
        if self.state.active:
            logger.warning("Simulation is already active")
            return

        if self._current_trips is None:
            await self.initialize_data()

        self.state.active = True
        self.state.save_checkpoint(self.platform_state_path)

        logger.info(f"Starting simulation from time {self.state.current_time}")
        self.simulation_task = asyncio.create_task(self._simulation_loop())

    async def pause_simulation(self) -> None:
        """Pause/resume the simulation."""
        self.state.active = not self.state.active
        self.state.save_checkpoint(self.platform_state_path)

        logger.info(f"Simulation {'resumed' if self.state.active else 'paused'}")

        if self.state.active and (self.simulation_task is None or self.simulation_task.done()):
            self.simulation_task = asyncio.create_task(self._simulation_loop())

    async def _simulation_loop(self) -> None:
        """Main simulation loop."""
        max_simulation_time = 36000  # 10 hours in seconds
        transition_time = 18000  # 5 hours for test data

        try:
            while self.state.active and self.state.current_time < max_simulation_time * 2:  # 20h total
                loop_start = time.time()

                # Check for dataset transition (test → rain after 10h)
                if self.state.current_time >= max_simulation_time and self.state.dataset == "test":
                    await self._handle_day_transition()

                # Get current batch of trips
                batch_trips, ground_truth = self.get_trip_batch(self.state.current_time, self.state.batch_size_minutes)

                if batch_trips:
                    # Trigger trip processing (this will be handled by main backend)
                    if self.on_metrics_update:
                        await self.on_metrics_update(batch_trips, ground_truth)

                # Advance simulation time
                self.state.current_time += self.state.batch_size_minutes * 60

                # Save state periodically
                if self.state.current_time % (5 * 60) == 0:  # Every 5 minutes
                    self.state.save_checkpoint(self.platform_state_path)

                # Maintain real-time tick rate
                elapsed = (time.time() - loop_start) * 1000
                sleep_time = max(0, self.tick_interval_ms - elapsed) / 1000
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("Simulation loop cancelled")
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
        finally:
            self.state.active = False
            self.state.save_checkpoint(self.platform_state_path)

    async def _handle_day_transition(self) -> None:
        """Handle transition from test to rain dataset."""
        logger.info("Day transition: switching from test to rain dataset")

        self._set_current_dataset("rain")
        self.state.save_checkpoint(self.platform_state_path)

        # Reset simulation time to start of rain data
        self.state.current_time = 36000

        # Send notification
        if self.on_notification:
            await self.on_notification(
                {
                    "type": "day_transition",
                    "message": "Weather changed to rain - concept drift expected",
                    "timestamp": time.time(),
                }
            )

    def get_status(self) -> dict:
        """Get current simulation status."""
        max_time = 72000  # 20 hours total
        progress = (self.state.current_time / max_time) * 100 if max_time > 0 else 0

        return {
            "active": self.state.active,
            "current_time": self.state.current_time,
            "dataset": self.state.dataset,
            "speed_multiplier": self.state.speed_multiplier,
            "progress_percent": min(100, progress),
        }

    async def shutdown(self) -> None:
        """Shutdown simulation gracefully."""
        if self.simulation_task:
            self.simulation_task.cancel()
            try:
                await self.simulation_task
            except asyncio.CancelledError:
                pass

        self.state.active = False
        self.state.save_checkpoint(self.platform_state_path)
        logger.info("Simulation engine shut down")
