"""Manages drift detectors for a specific ML task."""

import logging

import numpy as np
import pandas as pd
from river.drift import ADWIN, KSWIN, PageHinkley

from thesis.common.config import (
    ADWIN_DELTA_CANDIDATES,
    GRACE_PERIOD_SAMPLES,
    KSWIN_ALPHA_CANDIDATES,
    KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS,
    PAGE_HINKLEY_DELTA_CANDIDATES,
    PAGE_HINKLEY_THRESHOLD_CANDIDATES,
    SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS,
    SPC_MIN_STD,
    SPC_N_STD_CANDIDATES,
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


class ADWINWithGrace:
    """
    Wrapper for ADWIN detector with explicit grace period handling.

    Attributes:
        drift_detected (bool): Whether drift has been detected.
    """

    def __init__(self, delta: float, grace_period_samples: int) -> None:
        self._adwin: ADWIN = ADWIN(delta=delta)
        self._grace_period_samples: int = grace_period_samples
        self._samples_seen: int = 0
        self.drift_detected: bool = False

    def update(self, absolute_error: float) -> None:
        """
        Update drift detector with new absolute error value.

        Args:
            absolute_error (float): New absolute error value to process.
        """
        self._adwin.update(absolute_error)
        self._samples_seen += 1

        if self._samples_seen <= self._grace_period_samples:
            self.drift_detected = False
        else:
            self.drift_detected = self._adwin.drift_detected


class DetectorManager:
    """
    Manages drift detectors for a specific ML task.

    Attributes:
        drift_detectors (dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector]): Dictionary of all drift detectors.
    """

    def __init__(self, smoothing_window_samples: int) -> None:
        self._smoothing_window_samples = smoothing_window_samples
        self.drift_detectors: dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector] = {}

    def _smooth_errors(self, errors: np.ndarray) -> np.ndarray:
        """
        Apply rolling mean smoothing to errors.

        Args:
            errors (np.ndarray): Raw errors to smooth.

        Returns:
            np.ndarray: Smoothed errors using rolling mean.
        """
        return pd.Series(errors).rolling(window=self._smoothing_window_samples, min_periods=1).mean().to_numpy()

    def calibrate(self, absolute_errors: list[float] | np.ndarray) -> None:
        """
        Calibrate drift detectors on absolute errors.

        Args:
            absolute_errors (list[float] | np.ndarray): Absolute errors for calibration.
        """
        absolute_errors_smoothed = self._smooth_errors(absolute_errors)

        chosen_adwin_delta = None
        warmup_ignore = min(len(absolute_errors_smoothed) - 1, self._smoothing_window_samples)
        adwin_test_stride = 5 if len(absolute_errors_smoothed) // 5 > 1000 else 2
        calibration_sequence = (
            absolute_errors_smoothed[warmup_ignore::adwin_test_stride]
            if warmup_ignore > 0
            else absolute_errors_smoothed[::adwin_test_stride]
        )

        for delta in ADWIN_DELTA_CANDIDATES:
            detector = ADWIN(delta=delta)
            fired = False

            for error in calibration_sequence:
                detector.update(error)
                if detector.drift_detected:
                    fired = True
                    break

            if not fired:
                chosen_adwin_delta = delta
                break

        if chosen_adwin_delta is None:
            chosen_adwin_delta = ADWIN_DELTA_CANDIDATES[-1]

        adwin = ADWINWithGrace(delta=chosen_adwin_delta, grace_period_samples=GRACE_PERIOD_SAMPLES)

        chosen_page_hinkley_delta = None
        chosen_page_hinkley_threshold = None

        for delta in PAGE_HINKLEY_DELTA_CANDIDATES:
            for threshold in PAGE_HINKLEY_THRESHOLD_CANDIDATES:
                detector = PageHinkley(min_instances=1, delta=delta, threshold=threshold)
                fired = False

                for error in absolute_errors_smoothed:
                    detector.update(error)
                    if detector.drift_detected:
                        fired = True
                        break

                if not fired:
                    chosen_page_hinkley_delta = delta
                    chosen_page_hinkley_threshold = threshold
                    break

            if chosen_page_hinkley_delta is not None:
                break

        if chosen_page_hinkley_delta is None or chosen_page_hinkley_threshold is None:
            chosen_page_hinkley_delta = PAGE_HINKLEY_DELTA_CANDIDATES[-1]
            chosen_page_hinkley_threshold = PAGE_HINKLEY_THRESHOLD_CANDIDATES[-1]

        page_hinkley = PageHinkley(
            min_instances=GRACE_PERIOD_SAMPLES, delta=chosen_page_hinkley_delta, threshold=chosen_page_hinkley_threshold
        )

        chosen_kswin_alpha = None
        chosen_kswin_window_size = None
        chosen_kswin_stat_size = None

        for window_size, stat_size in KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS:
            if len(absolute_errors) < window_size + stat_size:
                continue

            for alpha in KSWIN_ALPHA_CANDIDATES:
                detector = KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)
                fired = False

                for i, error in enumerate(absolute_errors):
                    detector.update(error)
                    if i >= window_size and detector.drift_detected:
                        fired = True
                        break

                if not fired:
                    chosen_kswin_alpha = alpha
                    chosen_kswin_window_size = window_size
                    chosen_kswin_stat_size = stat_size
                    break

            if chosen_kswin_alpha is not None:
                break

        if chosen_kswin_alpha is None:
            chosen_kswin_alpha = KSWIN_ALPHA_CANDIDATES[-1]
            chosen_kswin_window_size = KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS[-1][0]
            chosen_kswin_stat_size = KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS[-1][1]

        kswin = KSWIN(alpha=chosen_kswin_alpha, window_size=chosen_kswin_window_size, stat_size=chosen_kswin_stat_size)

        errors_mean = np.mean(absolute_errors_smoothed)
        errors_std = np.std(absolute_errors_smoothed)
        chosen_spc_n_std = None
        chosen_spc_consecutive_violations_required = None

        for n_std in SPC_N_STD_CANDIDATES:
            error_threshold = errors_mean + n_std * (errors_std if errors_std > 0 else SPC_MIN_STD)

            current_consecutive_violations = 0
            maximum_consecutive_violations = 0
            for error in absolute_errors_smoothed:
                if error > error_threshold:
                    current_consecutive_violations += 1
                else:
                    maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)
                    current_consecutive_violations = 0
            maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)

            for consecutive_violations_required, multiplier in SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS:
                consecutive_violations_required = max(
                    consecutive_violations_required,
                    int(maximum_consecutive_violations * multiplier)
                    if maximum_consecutive_violations > 0
                    else consecutive_violations_required,
                )

                consecutive_violations = 0
                fired = False

                for error in absolute_errors_smoothed:
                    if error > error_threshold:
                        consecutive_violations += 1
                        if consecutive_violations >= consecutive_violations_required:
                            fired = True
                            break
                    else:
                        consecutive_violations = 0

                if not fired:
                    chosen_spc_n_std = n_std
                    chosen_spc_consecutive_violations_required = consecutive_violations_required
                    break

            if chosen_spc_n_std is not None:
                break

        if chosen_spc_n_std is None:
            chosen_spc_n_std = SPC_N_STD_CANDIDATES[-1]
            error_threshold = errors_mean + chosen_spc_n_std * (errors_std if errors_std > 0 else SPC_MIN_STD)

            current_consecutive_violations = 0
            maximum_consecutive_violations = 0
            for error in absolute_errors_smoothed:
                if error > error_threshold:
                    current_consecutive_violations += 1
                else:
                    maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)
                    current_consecutive_violations = 0
            maximum_consecutive_violations = max(maximum_consecutive_violations, current_consecutive_violations)

            consecutive_violations_required, multiplier = SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS[-1]
            chosen_spc_consecutive_violations_required = max(
                consecutive_violations_required,
                int(maximum_consecutive_violations * multiplier)
                if maximum_consecutive_violations > 0
                else consecutive_violations_required,
            )

        error_threshold = errors_mean + chosen_spc_n_std * (errors_std if errors_std > 0 else SPC_MIN_STD)

        spc = SPCDetector(
            error_threshold=error_threshold,
            consecutive_violations_required=chosen_spc_consecutive_violations_required,
            grace_period_samples=GRACE_PERIOD_SAMPLES,
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
