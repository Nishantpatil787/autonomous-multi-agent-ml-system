"""Logging utility for the Autonomous Multi-Agent Machine Learning System."""

import logging
import sys
from pathlib import Path
from typing import Optional

from utils.config import Config


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure and return a standardized logger instance.

    Args:
        name: Name of the module or agent requesting the logger.
        log_file: Optional path to a file where logs should also be written.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent re-adding handlers if logger is already configured
    if logger.hasHandlers():
        return logger

    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Standard console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
