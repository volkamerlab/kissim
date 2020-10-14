"""
kissim.utils

Useful utilities used across the library.
"""

import logging
import os
import shutil
import tempfile
import contextlib

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def enter_temp_directory(remove=True):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    os.chdir(temp_dir)
    _logger.debug("Entered %s", temp_dir)
    yield cwd, temp_dir
    os.chdir(cwd)
    _logger.debug("Left %s", temp_dir)
    if remove:
        _logger.debug("Deleting %s", temp_dir)
        shutil.rmtree(temp_dir)