"""
State management for simulation and drift tracking.
"""

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Persistent simulation state."""

    current_time: int = 0
    dataset: str = "test"  # "test" or "rain"
    active: bool = False
    speed_multiplier: int = 400  # 20h → 3min
    batch_size_minutes: int = 1

    def save_checkpoint(self, path: Path) -> None:
        """Save state to JSON file."""
        try:
            with open(path / "simulation.json", "w") as f:
                json.dump(asdict(self), f, indent=2)
            logger.debug(f"Saved simulation state: time={self.current_time}, dataset={self.dataset}")
        except Exception as e:
            logger.error(f"Failed to save simulation state: {e}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "SimulationState":
        """Load state from JSON file."""
        try:
            with open(path / "simulation.json") as f:
                data = json.load(f)
                logger.info(f"Loaded simulation state: time={data.get('current_time', 0)}")
                return cls(**data)
        except FileNotFoundError:
            logger.info("No simulation checkpoint found, starting fresh")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load simulation state: {e}")
            return cls()


@dataclass
class DriftState:
    """In-memory drift tracking state."""

    eta_status: str = "stable"  # stable|collecting|retraining|swapped
    fuel_status: str = "stable"
    stops_status: str = "stable"

    # Rolling error buffers for drift detection (maxlen for memory management)
    eta_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    fuel_errors: deque = field(default_factory=lambda: deque(maxlen=1000))
    stops_errors: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Rolling MAE calculation buffers (shorter for UI updates)
    eta_mae_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    fuel_mae_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    stops_mae_buffer: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_errors(
        self, eta_error: Optional[float] = None, fuel_error: Optional[float] = None, stops_error: Optional[float] = None
    ) -> None:
        """Add new errors to buffers."""
        if eta_error is not None:
            self.eta_errors.append(eta_error)
            self.eta_mae_buffer.append(eta_error)
        if fuel_error is not None:
            self.fuel_errors.append(fuel_error)
            self.fuel_mae_buffer.append(fuel_error)
        if stops_error is not None:
            self.stops_errors.append(stops_error)
            self.stops_mae_buffer.append(stops_error)

    def get_current_mae(self) -> Dict[str, Optional[float]]:
        """Calculate current MAE from buffers."""
        return {
            "eta": sum(self.eta_mae_buffer) / len(self.eta_mae_buffer) if self.eta_mae_buffer else None,
            "fuel": sum(self.fuel_mae_buffer) / len(self.fuel_mae_buffer) if self.fuel_mae_buffer else None,
            "stops": sum(self.stops_mae_buffer) / len(self.stops_mae_buffer) if self.stops_mae_buffer else None,
        }

    def get_drift_status(self) -> Dict[str, str]:
        """Get current drift status for all models."""
        return {"eta": self.eta_status, "fuel": self.fuel_status, "stops": self.stops_status}

    def save_status(self, path: Path) -> None:
        """Save drift status to JSON (for debugging/monitoring)."""
        try:
            status_data = {
                "drift_status": self.get_drift_status(),
                "current_mae": self.get_current_mae(),
                "buffer_sizes": {
                    "eta_errors": len(self.eta_errors),
                    "fuel_errors": len(self.fuel_errors),
                    "stops_errors": len(self.stops_errors),
                },
            }
            with open(path / "drift_status.json", "w") as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save drift status: {e}")


@dataclass
class ModelVersions:
    """Track active model versions."""

    eta_version: str = "stable"
    fuel_version: str = "stable"
    stops_version: str = "stable"

    def save(self, path: Path) -> None:
        """Save model versions to JSON."""
        try:
            with open(path / "model_versions.json", "w") as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")

    @classmethod
    def load(cls, path: Path) -> "ModelVersions":
        """Load model versions from JSON."""
        try:
            with open(path / "model_versions.json") as f:
                data = json.load(f)
                return cls(**data)
        except FileNotFoundError:
            logger.info("No model versions found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load model versions: {e}")
            return cls()
