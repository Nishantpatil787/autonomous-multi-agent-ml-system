"""Logging utility for consistent logging across the application."""

import logging
import sys
from pathlib import Path
from typing import Optional

from utils.config import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def setup_logger(name: str, log_file: Optional[str] = None, level: str = LOG_LEVEL) -> logging.Logger:
    """Set up a logger with consistent formatting.

    Args:
        name: Name of the logger (usually module name)
        log_file: Optional log file name. If None, logs to console only.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(LOGS_DIR) / log_file
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger for general use
default_logger = setup_logger("autonomous_ml")
