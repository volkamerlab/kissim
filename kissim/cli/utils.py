"""
kissim.cli.utils

Utilities for CLI commands.
"""

import logging
from pathlib import Path
import platform


def configure_logger(filename=None, level=logging.INFO):
    """
    Configure logging.

    Parameters
    ----------
    filename : str or None
        Path to log file.
    level : int
        Logging level (default: INFO).
    """

    # Get logger for both the kissim and opencadd package
    logger_kissim = logging.getLogger("kissim")
    logger_opencadd = logging.getLogger("opencadd")

    # Set logger levels
    logger_kissim.setLevel(level)
    logger_opencadd.setLevel(level)

    # Get formatter
    formatter = logging.Formatter(logging.BASIC_FORMAT)

    # Set a stream and a file handler
    s_handler = logging.StreamHandler()
    # Set formatting for these handlers
    s_handler.setFormatter(formatter)
    # Add both handlers to both loggers
    logger_kissim.addHandler(s_handler)
    logger_opencadd.addHandler(s_handler)

    # Set up file handler if
    # - log file is given
    # - we are not under Windows, since logging and multiprocessing do not work here
    #   see more details here: https://github.com/volkamerlab/kissim/pull/49
    if filename and platform.system() != "Windows":
        filename = Path(filename)
        f_handler = logging.FileHandler(filename.parent / f"{filename.stem}.log", mode="w")
        f_handler.setFormatter(formatter)
        logger_kissim.addHandler(f_handler)
        logger_opencadd.addHandler(f_handler)
