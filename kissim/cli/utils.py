"""
kissim.cli.utils

Utilities for CLI commands.
"""

import logging
from pathlib import Path


def configure_logger(filename, level=logging.INFO):
    """
    Configure logging.

    Parameters
    ----------
    filename : str
        Path to log file.
    level : int
        Logging level (default: INFO).
    """

    filename = Path(filename)

    # Get logger for both the kissim and opencadd package
    logger_kissim = logging.getLogger("kissim")
    logger_opencadd = logging.getLogger("opencadd")

    # Set logger levels
    logger_kissim.setLevel(level)
    logger_opencadd.setLevel(level)

    # Set a stream and a file handler
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename.parent / f"{filename.stem}.log", mode="w")

    # Set formatting for these handlers
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    s_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add both handlers to both loggers
    logger_kissim.addHandler(s_handler)
    logger_opencadd.addHandler(s_handler)
    logger_kissim.addHandler(f_handler)
    logger_opencadd.addHandler(f_handler)
