"""Utility functions for formatting."""

from thesis.common.enums import MLTask

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
        seconds = int(value)
        minutes = seconds // 60
        return f"{minutes:02d}m {seconds:02d}s"

    elif ml_task == MLTask.FUEL:
        liters = value / 1000
        return f"{liters:.2f}L"

    elif ml_task == MLTask.STOPS:
        return f"{int(value)}"
