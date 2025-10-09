"""Drift detection service for ML tasks."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from thesis.common.config import CONSENSUS_THRESHOLD, GRACE_PERIOD_SAMPLES, SMOOTHING_WINDOW_SAMPLES
from thesis.common.enums import DriftDetectorType, DriftState, MLTask
from thesis.common.schemas import DriftErrorsResponse, DriftResetResponse, ErrorPoint, RecalibrateResponse
from thesis.drift.services.detector_manager import DetectorManager

logger = logging.getLogger(__name__)


@dataclass
class DriftSnapshot:
    """
    Drift snapshot for a single ML task.

    Attributes:
        state (DriftState): Current drift state.
        start_timestamp (int): Timestamp when current state started.
        detector_manager (DetectorManager): DetectorManager instance for this task.
        error_history (deque[float]): Circular buffer for error smoothing.
        samples_seen (int): Number of samples processed.
        fired_detectors (set[DriftDetectorType]): Set of drift detector types that have fired.
        running_sum (float): Running sum of errors.
    """

    state: DriftState
    start_timestamp: int
    detector_manager: DetectorManager
    error_history: deque[float]
    samples_seen: int
    fired_detectors: set[DriftDetectorType]
    running_sum: float


class DriftService:
    """Drift detection service for ML tasks."""

    def __init__(self, misc_dir: Path) -> None:
        self._misc_dir: Path = misc_dir
        self._consensus_threshold: int = CONSENSUS_THRESHOLD
        self._smoothing_window: int = SMOOTHING_WINDOW_SAMPLES
        self._grace_period_samples: int = GRACE_PERIOD_SAMPLES
        self._lock: asyncio.Lock = asyncio.Lock()
        self._snapshots: dict[MLTask, DriftSnapshot] = {}

    async def _initialize_task(self, ml_task: MLTask) -> None:
        """
        Initialize drift detection for a specific ML task.

        Args:
            ml_task (MLTask): ML task to initialize.

        Raises:
            FileNotFoundError: If calibrated drift detectors not found.
            RuntimeError: If drift detectors loaded but empty.
        """
        detector_manager = DetectorManager(self._misc_dir, ml_task, self._smoothing_window)
        detector_manager.load()

        if not detector_manager.drift_detectors:
            raise RuntimeError(f"Drift detectors loaded but empty for {ml_task}")

        self._snapshots[ml_task] = DriftSnapshot(
            state=DriftState.STABLE,
            start_timestamp=0,
            detector_manager=detector_manager,
            error_history=deque(maxlen=self._smoothing_window),
            samples_seen=0,
            fired_detectors=set(),
            running_sum=0.0,
        )

        logger.info(f"Initialized drift detection for {ml_task}")

    def _append_and_smooth_error(self, snapshot: DriftSnapshot, error: float) -> float:
        """
        Append error to history and compute rolling mean.

        Args:
            snapshot (DriftSnapshot): Drift snapshot containing error history.
            error (float): Error value to add to the window.

        Returns:
            float: Smoothed error value.
        """

        if len(snapshot.error_history) == snapshot.error_history.maxlen:
            oldest = snapshot.error_history[0]
            snapshot.running_sum -= oldest

        snapshot.error_history.append(error)
        snapshot.running_sum += error
        snapshot.samples_seen += 1
        return snapshot.running_sum / len(snapshot.error_history)

    async def process_errors(self, ml_task: MLTask, error_points: list[ErrorPoint]) -> DriftErrorsResponse:
        """
        Process batch of errors for drift detection.

        Args:
            ml_task (MLTask): ML task to process errors for.
            error_points (list[ErrorPoint]): List of error points with timestamps.

        Returns:
            DriftErrorsResponse: Current drift state after processing errors.
        """
        async with self._lock:
            if ml_task not in self._snapshots:
                await self._initialize_task(ml_task)

            snapshot = self._snapshots[ml_task]

            if snapshot.state == DriftState.DRIFTED:
                return DriftErrorsResponse(
                    state=snapshot.state,
                    start_timestamp=snapshot.start_timestamp,
                )

            for error_point in error_points:
                smoothed_error = self._append_and_smooth_error(snapshot, error_point.error)

                in_grace_period = snapshot.samples_seen <= self._grace_period_samples

                drift_detectors = snapshot.detector_manager.drift_detectors
                for drift_detector_type, drift_detector in drift_detectors.items():
                    drift_detector.update(smoothed_error)

                    if in_grace_period:
                        continue

                    if drift_detector.drift_detected and drift_detector_type not in snapshot.fired_detectors:
                        snapshot.fired_detectors.add(drift_detector_type)
                        logger.info(
                            f"[{ml_task}] {drift_detector_type} fired at sample {snapshot.samples_seen}, timestamp {error_point.timestamp}"
                        )

                if len(snapshot.fired_detectors) >= self._consensus_threshold and snapshot.state == DriftState.STABLE:
                    snapshot.state = DriftState.DRIFTED
                    snapshot.start_timestamp = error_point.timestamp
                    logger.warning(
                        f"[{ml_task}] Consensus drift detected at sample {snapshot.samples_seen}, timestamp {error_point.timestamp}"
                    )
                    return DriftErrorsResponse(
                        state=snapshot.state,
                        start_timestamp=snapshot.start_timestamp,
                    )

            return DriftErrorsResponse(
                state=snapshot.state,
                start_timestamp=snapshot.start_timestamp,
            )

    async def get_state(self, ml_task: MLTask) -> DriftErrorsResponse:
        """
        Get current drift state for an ML task.

        Args:
            ml_task (MLTask): ML task to get state for.

        Returns:
            DriftErrorsResponse: Current drift state.
        """
        async with self._lock:
            if ml_task not in self._snapshots:
                await self._initialize_task(ml_task)

            snapshot = self._snapshots[ml_task]
            return DriftErrorsResponse(
                state=snapshot.state,
                start_timestamp=snapshot.start_timestamp,
            )

    async def reset_tasks(self, ml_tasks: list[MLTask]) -> DriftResetResponse:
        """
        Reset drift detection for a list of ML tasks.

        Args:
            ml_tasks (list[MLTask]): List of ML tasks to reset.

        Returns:
            DriftResetResponse: Response containing success status.
        """
        async with self._lock:
            for ml_task in ml_tasks:
                if ml_task in self._snapshots:
                    await self._initialize_task(ml_task)
                    logger.info(f"Reset drift detection for {ml_task}")

            return DriftResetResponse(success=True)

    async def recalibrate_task(self, ml_task: MLTask, post_adaptation_errors: list[float]) -> RecalibrateResponse:
        """
        Recalibrate drift detectors after model adaptation.

        Args:
            ml_task (MLTask): ML task to recalibrate.
            post_adaptation_errors (list[float]): Post-adaptation errors from retrained model.

        Returns:
            RecalibrateResponse: Recalibration success status.
        """
        async with self._lock:
            await self._initialize_task(ml_task)

            snapshot = self._snapshots[ml_task]
            detector_manager = snapshot.detector_manager
            detector_manager.calibrate(post_adaptation_errors)

            logger.info(f"Recalibrated drift detectors for {ml_task}")

            return RecalibrateResponse(success=True)

    async def clear(self) -> None:
        """Clear the drift service."""
        async with self._lock:
            for _, snapshot in self._snapshots.items():
                snapshot.detector_manager.clear()

            logger.info("Cleared drift service")
