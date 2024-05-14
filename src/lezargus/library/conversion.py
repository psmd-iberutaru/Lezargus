"""Functions to convert things into something else.

Any and all generic conversions (string, units, or otherwise) can be found in
here. Extremely standard conversion functions are welcome in here, but,
sometimes, a simple multiplication factor is more effective.
"""

import astropy.io.fits
import astropy.units

from lezargus.library import hint
from lezargus.library import logging


def convert_units(
    value: float | hint.ndarray,
    value_unit: hint.Unit | str,
    result_unit: hint.Unit | str,
) -> float | hint.ndarray:
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
        conversion_factor = value_unit.to(result_unit)
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


def parse_astropy_unit(unit_string: str | hint.Unit) -> hint.Unit:
    """Parse a unit string to an Astropy Unit class.

    Although for most cases, it is easier to use the Unit instantiation class
    directly, Astropy does not properly understand some unit conventions so
    we need to parse them in manually. Because of this, we just build a unified
    interface for all unit strings in general.

    Parameters
    ----------
    unit_string : str or Astropy Unit.
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
