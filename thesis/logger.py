import logging
import logging.handlers
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BACKEND_LOGGER_NAME = "backend"
DATASET_LOGGER_NAME = "dataset"
ETA_LOGGER_NAME = "eta"


def setup_logger(
    name: str,
    log_level: str = "DEBUG",
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.

    Args:
        name (str): Logger name.
        log_level (str): Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_file_size (int): Maximum size of log file in bytes before rotation.
        backup_count (int): Number of backup files to keep.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

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
