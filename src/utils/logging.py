"""Logging configuration and utilities."""

import logging
import sys


def get_logger(name: str = "RiskTrail", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger.

    Args:
        name: Name of the logger.
        level: Logging level (default INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
