"""Manages drift detectors for a specific ML task."""

import logging

import numpy as np
import pandas as pd
from river.drift import ADWIN, KSWIN, PageHinkley

from thesis.common.config import (
    ALPHA_CANDIDATES,
    DELTA_CANDIDATES,
    GRACE_PERIOD_SAMPLES,
    KSWIN_STAT_SIZE,
    KSWIN_WINDOW_SIZE,
    PAGE_HINKLEY_DELTA,
    SPC_CONSECUTIVE_VIOLATIONS_REQUIRED,
    SPC_MIN_STD,
    SPC_N_STD,
    THRESHOLD_CANDIDATES,
)
from thesis.common.enums import DriftDetectorType

logger = logging.getLogger(__name__)


class SPCDetector:
    """
    Statistical Process Control drift detector.

    Attributes:
        drift_detected (bool): Whether drift has been detected.
    """

    def __init__(self, error_threshold: float, consecutive_violations_required: int, grace_period_samples: int) -> None:
        self._error_threshold: float = error_threshold
        self._consecutive_violations_required: int = consecutive_violations_required
        self._grace_period_samples: int = grace_period_samples
        self._samples_seen: int = 0
        self._consecutive_violations: int = 0
        self.drift_detected: bool = False

    def update(self, absolute_error: float) -> None:
        """
        Update drift detector with new absolute error value.

        Args:
            absolute_error (float): New absolute error value to process.
        """
        self._samples_seen += 1
        if self._samples_seen <= self._grace_period_samples or self.drift_detected:
            return

        if absolute_error > self._error_threshold:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        if self._consecutive_violations >= self._consecutive_violations_required:
            self.drift_detected = True

    def __getstate__(self) -> dict[str, int | float | None]:
        """
        Get state for serialization.

        Returns:
            dict[str, int | float | None]: State for serialization.
        """
        return {
            "error_threshold": self._error_threshold,
            "consecutive_violations_required": self._consecutive_violations_required,
            "grace_period_samples": self._grace_period_samples,
        }

    def __setstate__(self, state: dict[str, int | float | None]) -> None:
        """
        Set state from deserialization.

        Args:
            state (dict[str, int | float | None]): State for deserialization.
        """
        self._error_threshold = state["error_threshold"]
        self._consecutive_violations_required = state["consecutive_violations_required"]
        self._grace_period_samples = state["grace_period_samples"]
        self._samples_seen = 0
        self._consecutive_violations = 0
        self.drift_detected = False

    def clear(self) -> None:
        """Clear the SPC driftdetector."""
        pass


class DetectorManager:
    """
    Manages drift detectors for a specific ML task.

    Attributes:
        drift_detectors (dict[DriftDetectorType, ADWIN | PageHinkley | KSWIN | SPCDetector]): Dictionary of all drift detectors.
    """

    def __init__(self, smoothing_window: int) -> None:
        self._smoothing_window = smoothing_window
        self.drift_detectors: dict[DriftDetectorType, ADWIN | PageHinkley | KSWIN | SPCDetector] = {}

    def _smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        Apply rolling mean smoothing to errors.

        Args:
            errors (np.ndarray): Raw errors to smooth.

        Returns:
            np.ndarray: Smoothed errors using rolling mean.
        """
        return pd.Series(errors).rolling(window=self._smoothing_window, min_periods=1).mean().to_numpy()

    def calibrate(self, absolute_errors: list[float] | np.ndarray) -> None:
        """
        Calibrate drift detectors on absolute errors.

        Args:
            absolute_errors (list[float] | np.ndarray): Absolute errors for calibration.
        """
        absolute_errors_smoothed = self._smooth_errors(absolute_errors)
        delta_candidates = DELTA_CANDIDATES
        chosen_delta = delta_candidates[0]

        for delta in delta_candidates:
            detector = ADWIN(delta=delta)
            fired = False

            for error in absolute_errors_smoothed:
                detector.update(error)
                if detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_delta = delta
                break

        adwin = ADWIN(delta=chosen_delta)
        logger.info(f"Calibrated ADWIN drift detector with delta={chosen_delta}")

        threshold_candidates = THRESHOLD_CANDIDATES
        chosen_threshold = threshold_candidates[0]

        for threshold in threshold_candidates:
            detector = PageHinkley(min_instances=1, delta=PAGE_HINKLEY_DELTA, threshold=threshold)
            fired = False

            for error in absolute_errors_smoothed:
                detector.update(error)
                if detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_threshold = threshold
                break

        page_hinkley = PageHinkley(
            min_instances=GRACE_PERIOD_SAMPLES, delta=PAGE_HINKLEY_DELTA, threshold=chosen_threshold
        )
        logger.info(f"Calibrated Page-Hinkley drift detector with threshold={chosen_threshold}")

        alpha_candidates = ALPHA_CANDIDATES
        chosen_alpha = alpha_candidates[0]

        for alpha in alpha_candidates:
            detector = KSWIN(alpha=alpha, window_size=KSWIN_WINDOW_SIZE, stat_size=KSWIN_STAT_SIZE)
            fired = False

            for i, error in enumerate(absolute_errors_smoothed):
                detector.update(error)
                if i >= detector.window_size and detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_alpha = alpha
                break

        kswin = KSWIN(alpha=chosen_alpha, window_size=KSWIN_WINDOW_SIZE, stat_size=KSWIN_STAT_SIZE)
        logger.info(f"Calibrated KSWIN drift detector with alpha={chosen_alpha}")

        errors_mean = np.mean(absolute_errors_smoothed)
        errors_std = np.std(absolute_errors_smoothed)
        error_threshold = errors_mean + SPC_N_STD * (errors_std if errors_std > 0 else SPC_MIN_STD)

        current_consecutive_violations = 0
        maximum_consecutive_violations = 0
        for error in absolute_errors_smoothed:
            if error > error_threshold:
                current_consecutive_violations += 1
            else:
                maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)
                current_consecutive_violations = 0

        maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)
        consecutive_violations_required = max(
            SPC_CONSECUTIVE_VIOLATIONS_REQUIRED, int(maximum_consecutive_violations * 20)
        )

        spc = SPCDetector(
            error_threshold=error_threshold,
            consecutive_violations_required=consecutive_violations_required,
            grace_period_samples=GRACE_PERIOD_SAMPLES,
        )
        logger.info(
            f"Calibrated SPC drift detector with error_threshold={error_threshold}, consecutive_violations_required={consecutive_violations_required}"
        )

        self.drift_detectors = {
            DriftDetectorType.ADWIN: adwin,
            DriftDetectorType.PAGE_HINKLEY: page_hinkley,
            DriftDetectorType.KSWIN: kswin,
            DriftDetectorType.SPC: spc,
        }

    def clear(self) -> None:
        """Clear the detector manager."""
        pass
