# logger.py
# Sets up a logger that writes to both the console and a log file.
# Every module imports get_logger() from here instead of using print().

import logging
import os
import sys
from datetime import datetime

from src.config import LOG_FILE, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger.
    - INFO and above go to console (clean output)
    - DEBUG and above go to logs/pipeline.log (full detail)

    Usage:
        from src.logger import get_logger
        log = get_logger(__name__)
        log.info("Processing 1M records...")
        log.warning("Sensor SENSOR_042 has missing values")
        log.error("Model file not found at path X")
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    logger = logging.getLogger(name)

    # Don't add handlers twice if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)

    # ── file handler ───────────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
