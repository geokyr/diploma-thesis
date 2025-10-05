"""Detector manager for creating and calibrating detectors."""

import numpy as np


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

    # TODO: add clear method
