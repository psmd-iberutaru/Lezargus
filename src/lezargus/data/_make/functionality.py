"""Common functionality to the making data functions.

There are not that many common functions, so a single module is fine.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

# isort: split

import glob
import os

import lezargus
from lezargus.library import logging


def find_data_filename(basename: str) -> str:
    """Find the full data filename provided its basename and conventions.

    We use this function to find the full filename of a data file. By
    convention, every datafile is separated into organizing folders which may
    change with development. However, the actual basenames, which by
    should be unique, should not. So, we can use that to be lazy and find the
    needed file.

    Parameters
    ----------
    basename : str
        The basename of the file which we are going to parse.

    Returns
    -------
    filename : str
        The full data file filename.

    """
    # Getting all of the files...
    data_glob_pattern = lezargus.library.path.merge_pathname(
        directory=[lezargus.config.INTERNAL_MODULE_DATA_FILE_DIRECTORY, "**"],
        filename="*",
        extension="*",
    )
    data_filename_list = glob.glob(data_glob_pattern, recursive=True)

    # Attempting to find the file...
    potential_filenames = []
    for filedex in data_filename_list:
        # Getting the basename to compare to.
        test_basename = lezargus.library.path.get_filename_with_extension(
            pathname=filedex,
        )
        # Windows and some Linux distributions don't care about case.
        basename_casefold = basename.casefold()
        test_basename = test_basename.casefold()
        if basename_casefold == test_basename:
            # This file is a potential match.
            potential_filenames.append(filedex)

    # There should only be one data filename...
    # Dumb stuff to satisfy the linter.
    too_many_files = 2
    if len(potential_filenames) == 0:
        # There was no matching data file.
        logging.error(
            error_type=logging.FileError,
            message=(
                f"Data file {basename} does not exist in the data file"
                " directories."
            ),
        )
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                f"Internal data file loading failed; basename {basename} does"
                " not point to a data file."
            ),
        )
        filename = "None"
    elif len(potential_filenames) == 1:
        # All good.
        found_filename = potential_filenames[0]
        filename = os.path.abspath(found_filename)
        return filename
    elif len(potential_filenames) >= too_many_files:
        # Too many files.
        logging.error(
            error_type=logging.FileError,
            message=f"Too many matching data files for {basename}",
        )
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                f"Internal data file loading failed; basename {basename} "
                "points to too many files.."
            ),
        )
        filename = "Too Many"
    else:
        # The code should not reach here.
        logging.critical(
            critical_type=logging.LogicFlowError,
            message=(
                "All of the cases for the potential data filenames should have"
                " been caught."
            ),
        )
        filename = "LogicFlowError"
    # All done.
    return filename
