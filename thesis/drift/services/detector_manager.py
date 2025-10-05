"""Detector manager for creating and calibrating detectors."""

from pathlib import Path

import joblib
import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley

from thesis.common.config import (
    ALPHA_CANDIDATES,
    DELTA_CANDIDATES,
    DETECTORS_FILENAME,
    GRACE_PERIOD_SAMPLES,
    KSWIN_STAT_SIZE,
    KSWIN_WINDOW_SIZE,
    PAGE_HINKLEY_DELTA,
    SPC_CONSECUTIVE_VIOLATIONS_REQUIRED,
    SPC_N_STD,
    THRESHOLD_CANDIDATES,
)
from thesis.common.enums import DetectorType, MLTask


class SPCDetector:
    """
    Statistical Process Control drift detector.

    Attributes:
        drift_detected (bool): Whether drift has been detected.
    """

    def __init__(self, n_std: int, consecutive_violations_required: int, grace_period_samples: int) -> None:
        self._n_std: int = n_std
        self._consecutive_violations_required: int = consecutive_violations_required
        self._grace_period_samples: int = grace_period_samples
        self._error_threshold: float | None = None
        self._samples_seen: int = 0
        self._consecutive_violations: int = 0
        self.drift_detected: bool = False

    def configure(self, smoothed_absolute_errors: list[float] | np.ndarray) -> None:
        """
        Configure detector with smoothed absolute error statistics.

        Args:
            smoothed_absolute_errors: Smoothed absolute errors to compute control limits.
        """
        errors_mean = np.mean(smoothed_absolute_errors)
        errors_std = np.std(smoothed_absolute_errors)

        if errors_std == 0.0:
            errors_std = 1e-6

        self._error_threshold = errors_mean + self._n_std * errors_std

    def update(self, absolute_error: float) -> None:
        """
        Update detector with new absolute error value.

        Args:
            absolute_error: New absolute error value to process.
        """
        self._samples_seen += 1
        if self._samples_seen <= self._grace_period_samples or self.drift_detected:
            return

        if self._error_threshold is None:
            raise RuntimeError("SPC not configured properly.")

        if absolute_error > self._error_threshold:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        if self._consecutive_violations >= self._consecutive_violations_required:
            self.drift_detected = True

    def __getstate__(self) -> dict[str, int | float | None]:
        """Get state for serialization.

        Returns:
            dict[str, int | float | None]: State for serialization.
        """
        return {
            "n_std": self._n_std,
            "consecutive_violations_required": self._consecutive_violations_required,
            "grace_period_samples": self._grace_period_samples,
            "error_threshold": self._error_threshold,
        }

    def __setstate__(self, state: dict[str, int | float | None]) -> None:
        """Set state from deserialization.

        Args:
            state (dict[str, int | float | None]): State for deserialization.
        """
        self._n_std = state["n_std"]
        self._consecutive_violations_required = state["consecutive_violations_required"]
        self._grace_period_samples = state["grace_period_samples"]
        self._error_threshold = state["error_threshold"]
        self._samples_seen = 0
        self._consecutive_violations = 0
        self.drift_detected = False

    def clear(self) -> None:
        """Clear the detector."""
        pass


class DetectorManager:
    """
    Manages drift detectors for a specific ML task.

    Attributes:
        detectors (dict[DetectorType, ADWIN | PageHinkley | KSWIN | SPCDetector]): Dictionary of all detectors.
    """

    def __init__(self, misc_dir: Path, ml_task: MLTask) -> None:
        self._detectors_path = misc_dir / ml_task / DETECTORS_FILENAME
        self.detectors: dict[DetectorType, ADWIN | PageHinkley | KSWIN | SPCDetector] = {}

    def calibrate(self, training_absolute_errors_smoothed: list[float] | np.ndarray) -> None:
        """
        Calibrate detectors on training errors.

        Args:
            training_absolute_errors_smoothed: Smoothed training absolute errors for calibration.
        """
        delta_candidates = DELTA_CANDIDATES
        chosen_delta = delta_candidates[0]

        for delta in delta_candidates:
            detector = ADWIN(delta=delta)
            fired = False

            for error in training_absolute_errors_smoothed:
                detector.update(error)
                if detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_delta = delta
                break

        adwin = ADWIN(delta=chosen_delta)

        threshold_candidates = THRESHOLD_CANDIDATES
        chosen_threshold = threshold_candidates[0]

        for threshold in threshold_candidates:
            detector = PageHinkley(min_instances=1, delta=PAGE_HINKLEY_DELTA, threshold=threshold)
            fired = False

            for error in training_absolute_errors_smoothed:
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

        alpha_candidates = ALPHA_CANDIDATES
        chosen_alpha = alpha_candidates[0]

        for alpha in alpha_candidates:
            detector = KSWIN(alpha=alpha, window_size=KSWIN_WINDOW_SIZE, stat_size=KSWIN_STAT_SIZE)
            fired = False

            for i, error in enumerate(training_absolute_errors_smoothed):
                detector.update(error)
                if i >= detector.window_size and detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_alpha = alpha
                break

        kswin = KSWIN(alpha=chosen_alpha, window_size=KSWIN_WINDOW_SIZE, stat_size=KSWIN_STAT_SIZE)

        spc = SPCDetector(
            n_std=SPC_N_STD,
            consecutive_violations_required=SPC_CONSECUTIVE_VIOLATIONS_REQUIRED,
            grace_period_samples=GRACE_PERIOD_SAMPLES,
        )
        spc.configure(training_absolute_errors_smoothed)

        self.detectors = {
            DetectorType.ADWIN: adwin,
            DetectorType.PAGE_HINKLEY: page_hinkley,
            DetectorType.KSWIN: kswin,
            DetectorType.SPC: spc,
        }

    def save(self) -> None:
        """Save current detectors."""
        joblib.dump(self.detectors, self._detectors_path)

    def load(self) -> None:
        """
        Load the detectors.

        Raises:
            FileNotFoundError: If the detectors file does not exist.
        """
        if not self._detectors_path.exists():
            raise FileNotFoundError(f"Detectors file not found: {self._detectors_path}")

        self.detectors = joblib.load(self._detectors_path)
