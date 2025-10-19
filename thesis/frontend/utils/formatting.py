"""Utility functions for formatting."""

from thesis.common.enums import DriftState, MLTask, NotificationLevel, ReportStatus, SimulationState

_ML_TASK_TITLES = {
    MLTask.ETA: "Estimated Time of Arrival",
    MLTask.FUEL: "Fuel Consumption",
    MLTask.STOPS: "Number of Stops",
}

_ML_TASK_UNITS = {
    MLTask.ETA: "seconds",
    MLTask.FUEL: "liters",
    MLTask.STOPS: "stops",
}

_ML_TASK_ICONS = {
    MLTask.ETA: "bi-clock-fill",
    MLTask.FUEL: "bi-fuel-pump-fill",
    MLTask.STOPS: "bi-stoplights-fill",
}

_DRIFT_STATE_COLORS = {
    DriftState.CALIBRATING: "info",
    DriftState.STABLE: "success",
    DriftState.DRIFTED: "danger",
    DriftState.RETRAINING: "warning",
}

_SIMULATION_STATE_COLORS = {
    SimulationState.READY: "info",
    SimulationState.RUNNING: "success",
    SimulationState.PAUSED: "warning",
    SimulationState.COMPLETED: "info",
}

_NOTIFICATION_LEVEL_COLORS = {
    NotificationLevel.INFO: "info",
    NotificationLevel.SUCCESS: "success",
    NotificationLevel.WARNING: "warning",
    NotificationLevel.DANGER: "danger",
}

_REPORT_STATUS_TOOLTIPS = {
    ReportStatus.NOT_STARTED: "AI Summary Report will be generated after the simulation completes",
    ReportStatus.GENERATING: "AI Summary Report is being generated",
    ReportStatus.FAILED: "AI Summary Report generation failed",
    ReportStatus.READY: "AI Summary Report has been generated",
}


def get_ml_task_title(ml_task: MLTask | str) -> str:
    """
    Get the full title of an ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Full title of the ML task.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _ML_TASK_TITLES[ml_task]


def get_ml_task_unit(ml_task: MLTask | str) -> str:
    """
    Get the unit of an ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Unit for the ML task.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _ML_TASK_UNITS[ml_task]


def get_ml_task_icon(ml_task: MLTask | str) -> str:
    """
    Get the icon of an ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Icon for the ML task.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _ML_TASK_ICONS[ml_task]


def get_drift_state_color(drift_state: DriftState | str) -> str:
    """
    Get the color of a drift state.

    Args:
        drift_state (DriftState | str): Drift state or string value.

    Returns:
        str: Color for the drift state.
    """
    if isinstance(drift_state, str):
        drift_state = DriftState(drift_state)

    return _DRIFT_STATE_COLORS[drift_state]


def get_simulation_state_color(simulation_state: SimulationState | str) -> str:
    """
    Get the color of a simulation state.

    Args:
        simulation_state (SimulationState | str): Simulation state or string value.

    Returns:
        str: Color for the simulation state.
    """
    if isinstance(simulation_state, str):
        simulation_state = SimulationState(simulation_state)

    return _SIMULATION_STATE_COLORS[simulation_state]


def get_notification_level_color(notification_level: NotificationLevel | str) -> str:
    """
    Get the color of a notification level.

    Args:
        notification_level (NotificationLevel | str): Notification level or string value.

    Returns:
        str: Color for the notification level.
    """
    if isinstance(notification_level, str):
        notification_level = NotificationLevel(notification_level)

    return _NOTIFICATION_LEVEL_COLORS[notification_level]


def get_report_status_tooltip(report_status: ReportStatus | str) -> str | None:
    """
    Get the tooltip message for a report status.

    Args:
        report_status (ReportStatus | str): Report status or string value.

    Returns:
        str | None: Tooltip message for the report status, or None if no tooltip needed.
    """
    if isinstance(report_status, str):
        report_status = ReportStatus(report_status)

    return _REPORT_STATUS_TOOLTIPS.get(report_status, None)


def format_simulation_timestamp(timestamp: int) -> str:
    """
    Format simulation timestamp.

    Args:
        timestamp (int): Simulation timestamp in seconds.

    Returns:
        str: Formatted timestamp string in "Day X HH:MM" format.
    """
    day = (timestamp // 36000) + 1
    remaining_seconds = timestamp % 36000
    hours = (remaining_seconds // 3600) + 8
    minutes = (remaining_seconds % 3600) // 60

    return f"Day {day:02d} - {hours:02d}:{minutes:02d}"


def format_prediction_value(ml_task: MLTask | str, value: float) -> str:
    """
    Format a prediction value based on the ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.
        value (float): Prediction value.

    Returns:
        str: Formatted prediction value with appropriate unit.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    if ml_task == MLTask.ETA:
        total = int(value)
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02d}:{seconds:02d} min"

    elif ml_task == MLTask.FUEL:
        liters = value / 740000
        return f"{liters:.2f} L"

    elif ml_task == MLTask.STOPS:
        stops = round(value)
        return f"{stops} stop{'' if stops == 1 else 's'}"
