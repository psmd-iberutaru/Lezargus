"""Make functions to create the generally non-configurable constants.

This module is created to making the near-non-configurable constants in the
data module. Configurable constants should of course be under the domain
of the configuration file.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


from lezargus.data._make import functionality
from lezargus.library import logging


def make_constant(
    key: str,
    basename: str = "constants.txt",
) -> hint.Any:
    """Load a single constant value from the main file, based on the key.

    Parameters
    ----------
    key : str
        The constant key value which we are going to be pulling from the
        constants file.
    basename : str, default = "constants.txt"
        The basename of the internal data file of the optic efficiency file.
        The paths are handled automatically. We default to the expected
        constant file.

    Returns
    -------
    constant_value : None | int | float | str
        The constant value.

    """
    # Parsing the filename.
    constants_filename = functionality.parse_basename(basename=basename)
    # Sanitizing the key input.
    clean_key = key.upper().strip()
    if not clean_key.startswith("CONST_"):
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Key {clean_key} does not begin with 'CONST_' and so is an"
                " invalid constant key."
            ),
        )

    # Opening the file.
    with open(constants_filename, encoding="utf8") as file:
        file_lines = file.readlines()

    # We need to find the line in the file which has the constant value.
    constant_line = None
    for linedex in file_lines:
        if linedex.startswith("#"):
            # Line is a comment line, skip.
            continue
        if linedex.startswith(clean_key):
            constant_line = linedex
            break
    # If the key failed to find.
    if constant_line is None:
        logging.error(
            error_type=logging.DevelopmentError,
            message=(
                f"Key {clean_key} does not match any entry in the constant"
                f" file: {constants_filename}."
            ),
        )
        str_key = clean_key
        str_value = "None"
    else:
        # Otherwise, we break the entry line into the actual entry.
        str_key, str_value = constant_line.split("=")
        str_key = str_key.strip()
        str_value = str_value.strip()
        # Last check.
        if str_key != clean_key:
            logging.critical(
                critical_type=logging.DevelopmentError,
                message=(
                    f"Input key {clean_key} matched constant line key"
                    f" {str_key}, but they are not equal."
                ),
            )

    # Finally, we need to convert it between one of the four types.
    if str_value.casefold() == "none" or str_value is None:
        constant_value = None
        return constant_value
    # We then attempt integers or floats.
    try:
        num_value = float(str_value)
    except ValueError:
        # The value is likely a string.
        constant_value = str(str_value)
    else:
        if num_value.is_integer():
            constant_value = int(num_value)
        else:
            constant_value = float(num_value)

    # All done.
    return constant_value
