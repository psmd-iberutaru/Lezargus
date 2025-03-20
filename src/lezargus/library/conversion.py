"""Functions to convert things into something else.

Any and all generic conversions (string, units, or otherwise) can be found in
here. Extremely standard conversion functions are welcome in here, but,
sometimes, a simple multiplication factor is more effective.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.io.fits
import astropy.units
import numpy as np

from lezargus.library import logging


def convert_units(
    value: float | hint.NDArray,
    value_unit: hint.Unit | str,
    result_unit: hint.Unit | str,
) -> float | hint.NDArray:
    """Convert a value from one unit to another unit.

    We convert values using Astropy, however, we only convert raw numbers and
    so we do not handle Quantity variables. The unit arguments are parsed
    with :py:func:`parse_astropy_unit` if it is not a unit. This function is
    vectorized properly, of course, as it is generally just multiplication.

    Parameters
    ----------
    value : float or ndarray
        The value to convert.
    value_unit : Unit or str
        The unit of the value we are converting. Parsing is attempted if it
        is not an Astropy Unit.
    result_unit : Unit or str
        The unit that we are converting to. Parsing is attempted if it
        is not an Astropy Unit.

    Returns
    -------
    result : float or ndarray
        The result after the unit conversion.

    """
    # We parse the units so we can use Astropy to do the unit conversions.
    value_unit = parse_astropy_unit(unit_string=value_unit)
    result_unit = parse_astropy_unit(unit_string=result_unit)

    # Determine the conversion factor and convert between the two.
    try:
        conversion_factor = float(value_unit.to(result_unit))
    except astropy.units.UnitConversionError as error:
        # The unit failed to convert. Astropy's message is actually pretty
        # informative so we bootstrap it.
        astropy_error_message = str(error)
        logging.critical(
            critical_type=logging.ArithmeticalError,
            message=f"Unit conversion failed: {astropy_error_message}",
        )
    # Applying the conversion.
    result = value * conversion_factor
    return result


def parse_numpy_dtype(dtype_string: str | type) -> type:
    """Parse a data type string to an Numpy data type.

    We only support built-in normal Python types and the Numpy types.
    Unsupported types will not be converted by this function, and will raise
    an error.

    Parameters
    ----------
    dtype_string : str
        The data type, either as a string representation (typical) or a type
        (for compatibility reasons). Only Numpy canonical names are accepted.

    Returns
    -------
    numpy_type : type
        The data type after the conversion.

    """
    # We need to determine the type. If a type has been provided, it is easy
    # to just go through all of the same process.
    if isinstance(dtype_string, type):
        dtype_string = dtype_string.__name__
    elif isinstance(dtype_string, str):
        # All good.
        pass
    else:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"Object of type {type(dtype_string)} is not supported for data"
                " type interpretation."
            ),
        )
    dtype_string = str(dtype_string).casefold()

    # To avoid PLR0912, and to make it easier to add new options, a full
    # dictionary is used instead.
    type_dictionary = {
        "int": int,
        "int64": np.int64,
        "int32": np.int32,
        # Floats...
        "float": float,
        "float64": np.float64,
        "float32": np.float32,
        "float16": np.float16,
        # Strings...
        "str": str,
        # Other things?
        "object": object,
    }

    # Getting the type. Note, because None itself is also a valid type,
    # we need to be more creative for the case of a missing entry.
    numpy_type = type_dictionary.get(dtype_string, logging.ExpectedCaughtError)
    if numpy_type == logging.ExpectedCaughtError:
        logging.error(
            error_type=logging.InputError,
            message=f"Data type string {dtype_string} not implemented.",
        )
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                f"Data type string {dtype_string} does not have a"
                " conversion, it could implemented."
            ),
        )

    # All done.
    return numpy_type


def parse_astropy_unit(unit_string: str | hint.Unit | None) -> hint.Unit:
    """Parse a unit string to an Astropy Unit class.

    Although for most cases, it is easier to use the Unit instantiation class
    directly, Astropy does not properly understand some unit conventions so
    we need to parse them in manually. Because of this, we just build a unified
    interface for all unit strings in general.

    Parameters
    ----------
    unit_string : str or Astropy Unit or None.
        The unit string to parse into an Astropy unit. If it is None, then we
        return a dimensionless quantity unit.

    Returns
    -------
    unit_instance : Unit
        The unit instance after parsing.

    """
    # If it is already a unit, just return it.
    if isinstance(unit_string, astropy.units.UnitBase):
        return unit_string

    # We check for a few input cases which Astropy does not natively know
    # but we do.
    # ...for dimensionless unit entries...
    unit_string = "" if unit_string is None else unit_string
    # ...for flams, the unit of spectral density over wavelength...
    unit_string = "erg / (AA cm^2 s)" if unit_string == "flam" else unit_string

    # Finally, converting the string.
    try:
        unit_instance = astropy.units.Unit(unit_string, parse_strict="raise")
    except ValueError:
        # The unit string provided is likely not something we can parse.
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Input unit string cannot be parsed to an Astropy unit"
                f" {unit_string}."
            ),
        )
    # All done.
    return unit_instance
