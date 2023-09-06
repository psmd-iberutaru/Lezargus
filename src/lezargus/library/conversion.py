"""Functions to convert things into something else.

Any and all generic conversions (string, units, or otherwise) can be found in
here. Extremely standard conversion functions are welcome in here, but,
sometimes, a simple multiplication factor is more effective.
"""

import astropy.io.fits
import astropy.units
import numpy as np

from lezargus.library import hint
from lezargus.library import logging


def parse_unit_to_astropy_unit(unit_string: str) -> hint.Unit:
    """Parse a unit string to an Astropy Unit class.

    Although for most cases, it is easier to use the Unit instantiation class
    directly, Astropy does not properly understand some unit conventions so
    we need to parse them in manually. Because of this, we just build a unified
    interface for all unit strings in general.

    Parameters
    ----------
    unit_string : str
        The unit string to parse into an Astropy unit. If it is None, then we
        return a dimensionless quantity unit.

    Returns
    -------
    unit_instance : Unit
        The unit instance after parsing.
    """
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
                "It is likely the unit string provided is not something that"
                " can be parsed into an Astropy unit:  {us}".format(
                    us=unit_string,
                )
            ),
        )
    # All done.
    return unit_instance


def convert_to_allowable_fits_header_data_types(
    input_data: object,
) -> str | int | float | bool | hint.Undefined:
    """Convert any input into something FITS headers allow.

    Per the FITS standard, the allowable data types which values entered in
    FITS headers is a subset of what Python can do. As such, this function
    converts any type of reasonable input into something the FITS headers
    would allow. Note, we mostly do basic checking and conversions. If the
    object is too exotic, it may cause issues down the line.

    In general, only strings, integers, floating point, boolean, and no values
    are allowed. Astropy usually will handle further conversion from the basic
    Python types so we only convert up to there.

    Parameters
    ----------
    input_data : object
        The input to convert into an allowable FITS header keyword.

    Returns
    -------
    header_output : str, int, float, bool, or None
        The output after conversion. Note the None is not actually a None
        type itself, but Astropy's header None/Undefined type.
    """
    # If it is None, then we assume a blank record.
    if input_data is None or isinstance(
        input_data,
        astropy.io.fits.card.Undefined,
    ):
        # By convention, this should be a blank record; Astropy has a nice
        # way of providing it.
        return astropy.io.fits.card.Undefined()

    # If it is an boolean or integer, it is fine as well.
    if isinstance(input_data, bool | int):
        # All good, again, returning just the basic type.
        return input_data

    # If the value is a floating point value, we need to check if it is an
    # actual number or not.
    if isinstance(input_data, float):
        # If it is an otherwise good number, it is valid, but if it is
        # not finite, we need to handle it appropriately.
        if np.isfinite(input_data):
            # All good.
            return float(input_data)
        # Infinites are not well represented and we use strings
        # instead. FITS also does not understand NaNs very well, so we just
        # use strings.
        if np.isinf(input_data) or np.isnan(input_data):
            return str(input_data)
        # If you get here, then the number is not a typical float.
        # We see if the future string conversion can deal with it.
        logging.warning(
            warning_type=logging.DataLossWarning,
            message=(
                "The header input value {val} is a float type, but is not a"
                " standard float type understandable by this conversion"
                " function.".format(val=input_data)
            ),
        )
    # We do not expect much use from complex numbers. The FITS standard does
    # not specify a fixed-format for complex numbers. We nevertheless check
    # and implement the format.
    if isinstance(input_data, complex):
        # We break it into its two parts per the standard and package it
        # as a string. Unfortunately, complex integer numbers are not really
        # supported in Python so transmitting it over is non-trivial. We
        # ignore this use case for now, complex floats should be good enough.
        return "({rl}, {im})".format(
            rl=np.real(input_data),
            im=np.imag(input_data),
        )

    # All Python objects can be strings, so we just cast it one to save.
    # However, if the string representation of the object is its __repr__, then
    # conversion really was not made and there is no way to save the data
    # without losing information.
    header_output = str(input_data)
    if header_output == repr(input_data):
        # A proper string conversion failed. We still return the representation
        # but the user should know of the loss of data.
        logging.warning(
            warning_type=logging.DataLossWarning,
            message=(
                "The input type {ty} cannot be properly cast into a one usable"
                " with FITS headers; only the __repr__ is used. Its value is:"
                " {val}.".format(
                    ty=type(input_data),
                    val=input_data,
                )
            ),
        )
    # All done.
    return header_output
