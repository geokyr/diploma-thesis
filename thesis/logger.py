import logging
import logging.handlers
import subprocess
import sys
from pathlib import Path


def setup_logger(
    experiment_name: str,
    logs_dir: Path,
    log_level: str = "DEBUG",
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup the root logger with console and file handlers and return a logger for the experiment.

    Args:
        experiment_name (str): The name of the experiment.
        logs_dir (Path): The directory to save the logs.
        log_level (str): Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_file_size (int): Maximum size of log file in bytes before rotation.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: A logger for the experiment.
    """
    root_logger = logging.getLogger()

    root_logger.setLevel(getattr(logging, log_level.upper()))

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file = logs_dir / f"{experiment_name}.log"
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(experiment_name)

    return logger


def log_subprocess_result(
    operation_name: str,
    logger: logging.Logger,
    command: list[str],
    result: subprocess.CompletedProcess,
) -> None:
    """
    Log a subprocess execution result with consistent formatting.

    Args:
        operation_name (str): Name of the operation for error messages.
        logger (logging.Logger): Logger instance to use.
        command (list[str]): The command that was executed.
        result (subprocess.CompletedProcess): The subprocess.CompletedProcess result.
    """
    command_str = " ".join(str(arg) for arg in command)
    logger.info(f"Executing {operation_name}")
    logger.info(f"Command: {command_str}")

    if result.stdout:
        logger.info(f"stdout from {operation_name}: {result.stdout}")

    if result.stderr:
        logger.warning(f"stderr from {operation_name}: {result.stderr}")
