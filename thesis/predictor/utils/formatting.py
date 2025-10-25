from thesis.common.enums import MLTask

_PREDICTOR_TITLE_MAP = {
    MLTask.ETA: "ETA",
    MLTask.FUEL: "Fuel",
    MLTask.STOPS: "Stops",
}


def get_predictor_title(ml_task: MLTask | str) -> str:
    """
    Get the title of a predictor of an ML task.

    Args:
        ml_task (MLTask | str): ML task enum or string value.

    Returns:
        str: Title of the predictor.
    """
    if isinstance(ml_task, str):
        ml_task = MLTask(ml_task)

    return _PREDICTOR_TITLE_MAP[ml_task]
