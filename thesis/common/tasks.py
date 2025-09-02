"""
Task definitions and port mapping for predictor services.
"""

from enum import StrEnum

from thesis.common.config import PORT_PREDICTOR_ETA, PORT_PREDICTOR_FUEL, PORT_PREDICTOR_STOPS


class Task(StrEnum):
    """
    Tasks.

    Attributes:
        ETA: ETA prediction task.
        FUEL: Fuel consumption prediction task.
        STOPS: Number of stops prediction task.
    """

    ETA = "eta"
    FUEL = "fuel"
    STOPS = "stops"


TASK_PORTS = {
    Task.ETA: PORT_PREDICTOR_ETA,
    Task.FUEL: PORT_PREDICTOR_FUEL,
    Task.STOPS: PORT_PREDICTOR_STOPS,
}


def get_port_by_task(task: Task) -> int:
    """
    Get the port number for a given task.

    Args:
        task (Task): Task.

    Returns:
        int: Port number for the task.

    Raises:
        ValueError: If task is not recognized.
    """
    if task not in TASK_PORTS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {list(Task)}")

    return TASK_PORTS[task]
