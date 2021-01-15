"""
kissim.cli.utils

Utilities for CLI commands.
"""

import logging
from pathlib import Path


def configure_logger(filename, level=logging.INFO):
    """
    Configure logging.

    TODO also include opencadd?
    """

    filename = Path(filename)
    logger = logging.getLogger("kissim")
    logger.setLevel(level)
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename.parent / f"{filename.stem}.log")
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
