"""Collection of a wide variety of data and input sanitization methods.

Data and input sanitization is important to make sure it matches the
program expectations and to make sure that processes do not encounter garbage
in to have garbage out.

Specific internal sanitization methods may be stored in the same module as the
internal subject instead of this one.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.io.fits
import numpy as np

from lezargus.library import logging


def clean_finite_arrays(*arrays: hint.NDArray) -> tuple[hint.NDArray]:
    """Return parallel arrays with any non-finite number removed from them.

    We remove all parallel-aligned values (aligned with each other) which are
    not a finite number, such as NaN and infinity. Because we remove data,
    the shape of the output arrays will likely be very different to the input.

    Parameters
    ----------
    *arrays : ndarray
        The arrays, which are all parallel, to remove the non-finite numbers
        from.

    Returns
    -------
    clean_arrays : tuple
        The cleaned arrays, arranged in a tuple, in the exact order they were
        input in as `arrays`.

    """
    # We need to make sure each array is compatible with themselves. We assume
    # the first array is the reference array for size and shape comparison.
    reference_array = arrays[0]
    sized_arrays = []
    for index, arraydex in enumerate(arrays):
        compatible, compatible_array = verify_broadcastability(
            reference_array=reference_array,
            test_array=arraydex,
        )
        # We skip the non-compatible arrays.
        if compatible:
            sized_arrays.append(compatible_array)
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Input array index {index} shape {arraydex.shape} is not"
                    " compatible with the first array reference shape of"
                    f" {reference_array.shape}. Skipping."
                ),
            )

    # We now find the aligned clean index of all of the arrays.
    clean_index = np.full_like(reference_array, True, dtype=bool)
    for arraydex in sized_arrays:
        clean_index = clean_index & np.isfinite(arraydex, dtype=bool)

    # Finally, only returning the cleaned arrays.
    clean_arrays = tuple(arraydex[clean_index] for arraydex in sized_arrays)
    return clean_arrays


def fix_fits_header_value(
    input_data: object,
) -> str | int | float | bool | hint.Undefined:
    """Fix any input into something FITS headers allow.

    Per the FITS standard, the allowable data types which values entered in
    FITS headers is a subset of what Python can do. As such, this function
    converts any type of reasonable input into something the FITS headers
    would allow. Note, we mostly do basic checking and conversions. If the
    object is too exotic, it may cause issues down the line.

    In general, only strings, integers, floating point, boolean, and blank
    values are allowed. Astropy usually will handle further conversion from
    the basic Python types so we only convert up to there.

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
                f"The header input value {input_data} is a float type, but is"
                " not a standard float type understandable by this conversion"
                " function."
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
        return f"({np.real(input_data)}, {np.imag(input_data)})"

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
                f"The input type {type(input_data)} cannot be properly cast"
                " into a one usable with FITS headers; only the __repr__ is"
                f" used. Its value is: {input_data}."
            ),
        )
    # All done.
    return header_output


def verify_broadcastability(
    reference_array: hint.NDArray,
    test_array: hint.NDArray,
) -> tuple[bool, hint.NDArray | None]:
    """Verify if a test array is broadcastable with the reference array.

    This function serves to see if two arrays are compatible in shape. If
    the "test" array is just a single number, we allow it to broadcast, and
    return it if needed.

    Parameters
    ----------
    reference_array : ndarray
        The reference array which we are testing against.
    test_array : ndarray
        The test array that we are testing to.

    Returns
    -------
    verify : bool
        The verification.
    broadcast : ndarray
        The broadcasted array, may be trashed if not needed,

    """
    # We need to make them like arrays.
    reference_array = np.array(reference_array)
    test_array = np.array(test_array)
    # We assume the verification is False.
    verify = False
    # The basic check.
    if reference_array.shape == test_array.shape:
        verify = True
        broadcast = test_array
    # The next check is if the parameter is a single value which we can fill
    # into a new array.
    elif (isinstance(test_array, np.ndarray) and test_array.size == 1) or (
        isinstance(test_array, np.number | float | int)
    ):
        # It is considered broadcast-able.
        verify = True
        broadcast = np.full_like(reference_array, test_array)
    else:
        # The verification did not return a good result.
        verify = False
        broadcast = None
    # All done.
    return verify, broadcast
