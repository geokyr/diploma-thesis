import logging
import logging.handlers
import sys
from pathlib import Path

from thesis.common.config import BACKUP_COUNT, MAX_FILE_SIZE


def setup_logger(
    logger_name: str,
    logs_dir: Path,
    log_level: str = "INFO",
    max_file_size: int = MAX_FILE_SIZE,
    backup_count: int = BACKUP_COUNT,
) -> logging.Logger:
    """
    Setup the root logger with console and file handlers and return a configured logger.

    Args:
        logger_name (str): Name of the logger.
        logs_dir (Path): Directory to save the logs.
        log_level (str): Minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_file_size (int): Maximum size of log file in bytes before rotation.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: Configured logger with console and file handlers.
    """
    log_level = getattr(logging, log_level.upper())
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file = logs_dir / f"{logger_name}.log"
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_file_size, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(logger_name)

    return logger
