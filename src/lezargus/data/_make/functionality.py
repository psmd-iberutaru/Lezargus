"""Common functionality to the making data functions.

There are not that many common functions, so a single module is fine.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

# isort: split

import os

import lezargus
from lezargus.library import logging


def parse_basename(basename: str) -> str:
    """Parse and expand the basename to the full path of the data file.

    Parameters
    ----------
    basename : str
        The basename of the file which we are going to parse.

    Returns
    -------
    filename : str
        The full data file filename.

    """
    # We parse the filename.
    filename = lezargus.library.path.merge_pathname(
        directory=lezargus.config.INTERNAL_MODULE_DATA_FILE_DIRECTORY,
        filename=basename,
        extension=None,
    )
    # As a double check, we make sure the data file exists.
    if not os.path.isfile(filename):
        logging.error(
            error_type=logging.FileError,
            message=f"Data file {filename} does not exist.",
        )
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                f"Internal data file loading failed; basename {basename} does"
                " not point to a data file."
            ),
        )
    # All done.
    return filename
