"""Manages drift detectors for a specific ML task."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from river.drift import ADWIN, KSWIN, PageHinkley

from thesis.common.config import (
    ADWIN_DELTA_CANDIDATES,
    GRACE_PERIOD_SAMPLES,
    HIGH_STRIDE,
    KSWIN_ALPHA_CANDIDATES,
    KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS,
    LOW_STRIDE,
    PAGE_HINKLEY_DELTA_CANDIDATES,
    PAGE_HINKLEY_THRESHOLD_CANDIDATES,
    SMOOTHING_MIN_PERIODS,
    SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS,
    SPC_MIN_STD,
    SPC_N_STD_CANDIDATES,
)
from thesis.common.enums import DriftDetectorType


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """
    Calibration parameters for drift detectors.

    Attributes:
        adwin_delta (float): ADWIN delta parameter.
        page_hinkley_delta (float): PageHinkley delta parameter.
        page_hinkley_threshold (float): PageHinkley threshold parameter.
        kswin_alpha (float): KSWIN alpha parameter.
        kswin_window_size (int): KSWIN window size parameter.
        kswin_stat_size (int): KSWIN stat size parameter.
        spc_error_threshold (float): SPC error threshold parameter.
        spc_consecutive_violations_required (int): SPC consecutive violations required parameter.
    """

    adwin_delta: float
    page_hinkley_delta: float
    page_hinkley_threshold: float
    kswin_alpha: float
    kswin_window_size: int
    kswin_stat_size: int
    spc_error_threshold: float
    spc_consecutive_violations_required: int


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


def smooth_errors(errors: list[float] | np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply rolling mean smoothing to errors.

    Args:
        errors (list[float] | np.ndarray): Raw errors to smooth.
        window_size (int): Size of the rolling window.

    Returns:
        np.ndarray: Smoothed errors using rolling mean.
    """
    return pd.Series(errors).rolling(window=window_size, min_periods=SMOOTHING_MIN_PERIODS).mean().to_numpy()


def _calibrate_adwin(absolute_errors_smoothed: np.ndarray, smoothing_window_samples: int) -> float:
    """
    Calibrate ADWIN detector.

    Args:
        absolute_errors_smoothed (np.ndarray): Smoothed absolute errors.
        smoothing_window_samples (int): Window size for error smoothing.

    Returns:
        float: Chosen ADWIN delta parameter.
    """
    warmup_ignore = min(len(absolute_errors_smoothed) - 1, smoothing_window_samples)
    test_stride = HIGH_STRIDE if len(absolute_errors_smoothed) > 5000 else LOW_STRIDE
    test_errors = absolute_errors_smoothed[warmup_ignore::test_stride]

    for delta in ADWIN_DELTA_CANDIDATES:
        detector = ADWIN(delta=delta)
        fired = False

        for error in test_errors:
            detector.update(error)
            if detector.drift_detected:
                fired = True
                break

        if not fired:
            return delta

    return ADWIN_DELTA_CANDIDATES[-1]


def _calibrate_page_hinkley(absolute_errors_smoothed: np.ndarray) -> tuple[float, float]:
    """
    Calibrate Page-Hinkley detector.

    Args:
        absolute_errors_smoothed (np.ndarray): Smoothed absolute errors.

    Returns:
        tuple[float, float]: Chosen delta and threshold parameters.
    """
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
                return delta, threshold

    return PAGE_HINKLEY_DELTA_CANDIDATES[-1], PAGE_HINKLEY_THRESHOLD_CANDIDATES[-1]


def _calibrate_kswin(absolute_errors: np.ndarray) -> tuple[float, int, int]:
    """
    Calibrate KSWIN detector.

    Args:
        absolute_errors (np.ndarray): Raw absolute errors (not smoothed).

    Returns:
        tuple[float, int, int]: Chosen alpha, window_size, and stat_size parameters.
    """
    test_stride = HIGH_STRIDE if len(absolute_errors) > 5000 else LOW_STRIDE
    test_errors = absolute_errors[::test_stride]

    for window_size, stat_size in KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS:
        if len(absolute_errors) < window_size + stat_size:
            continue

        for alpha in KSWIN_ALPHA_CANDIDATES:
            detector = KSWIN(alpha=alpha, window_size=window_size, stat_size=stat_size)
            fired = False

            for index, error in enumerate(test_errors):
                detector.update(error)
                if index * test_stride >= window_size and detector.drift_detected:
                    fired = True
                    break

            if not fired:
                return alpha, window_size, stat_size

    return (
        KSWIN_ALPHA_CANDIDATES[-1],
        KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS[-1][0],
        KSWIN_WINDOW_SIZE_STAT_SIZE_CONFIGS[-1][1],
    )


def _calibrate_spc(absolute_errors_smoothed: np.ndarray) -> tuple[float, int]:
    """
    Calibrate SPC detector.

    Args:
        absolute_errors_smoothed (np.ndarray): Smoothed absolute errors.

    Returns:
        tuple[float, int]: Chosen error_threshold and consecutive_violations_required parameters.
    """
    errors_mean = np.mean(absolute_errors_smoothed)
    errors_std = np.std(absolute_errors_smoothed)

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
                return error_threshold, consecutive_violations_required

    return (
        errors_mean + SPC_N_STD_CANDIDATES[-1] * (errors_std if errors_std > 0 else SPC_MIN_STD),
        SPC_CONSECUTIVE_VIOLATIONS_REQUIRED_MULTIPLIER_CONFIGS[-1][0],
    )


def compute_calibration_parameters(
    absolute_errors: list[float] | np.ndarray, smoothing_window_samples: int
) -> CalibrationParameters:
    """
    Compute calibration parameters for drift detectors.

    Args:
        absolute_errors (list[float] | np.ndarray): Absolute errors for calibration.
        smoothing_window_samples (int): Window size for error smoothing.

    Returns:
        CalibrationParameters: Calibration parameters for drift detectors.
    """
    absolute_errors = np.array(absolute_errors) if isinstance(absolute_errors, list) else absolute_errors

    absolute_errors_smoothed = smooth_errors(absolute_errors, smoothing_window_samples)

    chosen_adwin_delta = _calibrate_adwin(absolute_errors_smoothed, smoothing_window_samples)
    chosen_page_hinkley_delta, chosen_page_hinkley_threshold = _calibrate_page_hinkley(absolute_errors_smoothed)
    chosen_kswin_alpha, chosen_kswin_window_size, chosen_kswin_stat_size = _calibrate_kswin(absolute_errors)
    chosen_spc_error_threshold, chosen_spc_consecutive_violations_required = _calibrate_spc(absolute_errors_smoothed)

    return CalibrationParameters(
        adwin_delta=chosen_adwin_delta,
        page_hinkley_delta=chosen_page_hinkley_delta,
        page_hinkley_threshold=chosen_page_hinkley_threshold,
        kswin_alpha=chosen_kswin_alpha,
        kswin_window_size=chosen_kswin_window_size,
        kswin_stat_size=chosen_kswin_stat_size,
        spc_error_threshold=chosen_spc_error_threshold,
        spc_consecutive_violations_required=chosen_spc_consecutive_violations_required,
    )


def create_detectors_from_calibration_parameters(
    calibration_parameters: CalibrationParameters,
) -> dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector]:
    """
    Create drift detector instances from calibration parameters.

    Args:
        calibration_parameters (CalibrationParameters): Pre-computed calibration parameters.

    Returns:
        dict[DriftDetectorType, ADWINWithGrace | PageHinkley | KSWIN | SPCDetector]: Dictionary mapping detector types to detector instances.
    """
    adwin = ADWINWithGrace(delta=calibration_parameters.adwin_delta, grace_period_samples=GRACE_PERIOD_SAMPLES)

    page_hinkley = PageHinkley(
        min_instances=GRACE_PERIOD_SAMPLES,
        delta=calibration_parameters.page_hinkley_delta,
        threshold=calibration_parameters.page_hinkley_threshold,
    )

    kswin = KSWIN(
        alpha=calibration_parameters.kswin_alpha,
        window_size=calibration_parameters.kswin_window_size,
        stat_size=calibration_parameters.kswin_stat_size,
    )

    spc = SPCDetector(
        error_threshold=calibration_parameters.spc_error_threshold,
        consecutive_violations_required=calibration_parameters.spc_consecutive_violations_required,
        grace_period_samples=GRACE_PERIOD_SAMPLES,
    )

    return {
        DriftDetectorType.ADWIN: adwin,
        DriftDetectorType.PAGE_HINKLEY: page_hinkley,
        DriftDetectorType.KSWIN: kswin,
        DriftDetectorType.SPC: spc,
    }
