"""Drift detection worker."""

import asyncio
import logging
import queue
from collections import deque
from dataclasses import dataclass
from multiprocessing import Process, Queue

from river.drift import KSWIN, PageHinkley

from thesis.common.config import (
    CALIBRATION_WINDOW_SAMPLES,
    CONSENSUS_THRESHOLD,
    HIGH_STRIDE,
    SMOOTHING_WINDOW_SAMPLES,
)
from thesis.common.enums import DriftDetectorType, DriftState, MLTask
from thesis.common.schemas import DriftErrorsResponse, ErrorPoint
from thesis.drift.utils.detectors import (
    ADWINWithGrace,
    SPCDetector,
    compute_calibration_parameters,
    create_detectors_from_calibration_parameters,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftSnapshot:
    """
    Drift snapshot for a single ML task.

    Attributes:
        state (DriftState): Current drift state.
        start_timestamp (int): Timestamp when current state started.
        drift_detectors (dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector]): Dictionary of drift detector instances for this task.
        error_history (deque[float]): Circular buffer for error smoothing.
        samples_seen (int): Number of samples processed.
        fired_detectors (set[DriftDetectorType]): Set of drift detector types that have fired.
        running_sum (float): Running sum of errors.
        calibration_errors (list[float]): Collected errors during calibration phase.
        calibration_window_samples (int): Required samples for calibration.
        is_calibrating (bool): Flag to prevent concurrent calibration.
    """

    state: DriftState
    start_timestamp: int
    drift_detectors: dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector]
    error_history: deque[float]
    samples_seen: int
    fired_detectors: set[DriftDetectorType]
    running_sum: float
    calibration_errors: list[float]
    calibration_window_samples: int
    is_calibrating: bool


def initialize_snapshot() -> DriftSnapshot:
    """
    Initialize a drift snapshot with default values.

    Returns:
        DriftSnapshot: Initialized snapshot with default values.
    """
    return DriftSnapshot(
        state=DriftState.CALIBRATING,
        start_timestamp=0,
        drift_detectors={},
        error_history=deque(maxlen=SMOOTHING_WINDOW_SAMPLES),
        samples_seen=0,
        fired_detectors=set(),
        running_sum=0.0,
        calibration_errors=[],
        calibration_window_samples=CALIBRATION_WINDOW_SAMPLES,
        is_calibrating=False,
    )


class DriftWorker:
    """Dedicated worker for handling drift detection for one ML task."""

    def __init__(self, ml_task: MLTask):
        self._ml_task = ml_task
        self._request_queue = Queue()
        self._response_queue = Queue()
        self._process = None

    def start(self):
        """Start the worker."""
        self._process = Process(target=self._run, name=f"drift-worker-{self._ml_task}", daemon=True)
        self._process.start()

    def _run(self):
        """Main worker loop that maintains state and processes requests."""
        snapshot = initialize_snapshot()

        logger.info(f"[{self._ml_task}] Worker started")

        while True:
            try:
                request = self._request_queue.get()

                if request is None:
                    logger.info(f"[{self._ml_task}] Worker stopped")
                    break

                if request == "RESET":
                    snapshot = initialize_snapshot()
                    logger.info(f"[{self._ml_task}] Worker resetted")
                    self._response_queue.put(True)
                    continue

                response = self._process_errors(snapshot, request)
                self._response_queue.put(response)

            except queue.Empty:
                continue

    def _process_errors(self, snapshot: DriftSnapshot, error_points: list[ErrorPoint]) -> DriftErrorsResponse:
        """
        Process error batch and update state.

        Args:
            snapshot (DriftSnapshot): Current snapshot of drift state.
            error_points (list[ErrorPoint]): List of error points to process.

        Returns:
            DriftErrorsResponse: Current drift state after processing errors.
        """
        if snapshot.state == DriftState.DRIFTED:
            return DriftErrorsResponse(
                state=snapshot.state,
                start_timestamp=snapshot.start_timestamp,
            )

        if snapshot.state == DriftState.CALIBRATING:
            if snapshot.is_calibrating:
                return DriftErrorsResponse(
                    state=snapshot.state,
                    start_timestamp=snapshot.start_timestamp,
                )

            for error_point in error_points:
                snapshot.calibration_errors.append(error_point.error)
                snapshot.samples_seen += 1

                if len(snapshot.calibration_errors) >= snapshot.calibration_window_samples:
                    snapshot.is_calibrating = True

                    calibration_parameters = compute_calibration_parameters(
                        snapshot.calibration_errors, SMOOTHING_WINDOW_SAMPLES
                    )
                    drift_detectors = create_detectors_from_calibration_parameters(calibration_parameters)

                    snapshot.state = DriftState.STABLE
                    snapshot.start_timestamp = error_point.timestamp
                    snapshot.drift_detectors = drift_detectors
                    snapshot.error_history.clear()
                    snapshot.samples_seen = 0
                    snapshot.fired_detectors = set()
                    snapshot.running_sum = 0.0
                    snapshot.calibration_errors = []
                    snapshot.is_calibrating = False

                    logger.info(f"[{self._ml_task}] Calibration completed")
                    break

            if snapshot.state == DriftState.CALIBRATING:
                return DriftErrorsResponse(
                    state=snapshot.state,
                    start_timestamp=snapshot.start_timestamp,
                )

        for index, error_point in enumerate(error_points):
            smoothed_error = self._append_and_smooth_error(snapshot, error_point.error)

            for drift_detector_type, drift_detector in snapshot.drift_detectors.items():
                if drift_detector_type not in snapshot.fired_detectors:
                    if drift_detector_type != DriftDetectorType.KSWIN:
                        drift_detector.update(smoothed_error)
                    elif drift_detector_type == DriftDetectorType.KSWIN and index % HIGH_STRIDE == 0:
                        drift_detector.update(error_point.error)

                    if drift_detector.drift_detected:
                        snapshot.fired_detectors.add(drift_detector_type)
                        logger.info(
                            f"[{self._ml_task}] {drift_detector_type} fired at timestamp {error_point.timestamp} (sample {snapshot.samples_seen})"
                        )

            if len(snapshot.fired_detectors) >= CONSENSUS_THRESHOLD:
                snapshot.state = DriftState.DRIFTED
                snapshot.start_timestamp = error_point.timestamp
                logger.info(
                    f"[{self._ml_task}] Consensus drift detected at timestamp {error_point.timestamp} (sample {snapshot.samples_seen})"
                )
                break

        return DriftErrorsResponse(
            state=snapshot.state,
            start_timestamp=snapshot.start_timestamp,
        )

    def _append_and_smooth_error(self, snapshot: DriftSnapshot, error: float) -> float:
        """
        Append error and compute rolling mean.

        Args:
            snapshot (DriftSnapshot): Current snapshot of drift state.
            error (float): Error to append and smooth.

        Returns:
            float: Smoothed error.
        """
        if len(snapshot.error_history) == snapshot.error_history.maxlen:
            oldest = snapshot.error_history[0]
            snapshot.running_sum -= oldest

        snapshot.error_history.append(error)
        snapshot.running_sum += error
        snapshot.samples_seen += 1
        return snapshot.running_sum / len(snapshot.error_history)

    async def submit(self, error_points: list[ErrorPoint]) -> DriftErrorsResponse:
        """
        Submit error batch to worker process and return response asynchronously.

        Args:
            error_points (list[ErrorPoint]): List of error points to process.

        Returns:
            DriftErrorsResponse: Current drift state after processing errors.
        """
        self._request_queue.put(error_points)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._response_queue.get)

    async def reset(self) -> bool:
        """
        Reset worker state.

        Returns:
            bool: True if reset was successful, False otherwise.
        """
        self._request_queue.put("RESET")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._response_queue.get)

    def clear(self):
        """Clear the drift worker."""
        if self._process and self._process.is_alive():
            self._request_queue.put(None)
            self._process.join()
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
