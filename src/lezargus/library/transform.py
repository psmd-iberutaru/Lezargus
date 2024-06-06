"""Array or image transformations, typically affine transformations.

The transform of images and arrays are important, and here we separate many
similar functions into this module.
"""

import numpy as np
import scipy.ndimage

from lezargus.library import hint
from lezargus.library import logging


def translate_2d(
    array: hint.ndarray,
    x_shift: float,
    y_shift: float,
    mode: str = "constant",
    constant: float = np.nan,
) -> hint.ndarray:
    """Translate a 2D image array.

    This function is a convenient wrapper around Scipy's function.

    Parameters
    ----------
    array : ndarray
        The input array to be translated.
    x_shift : float
        The number of pixels that the array is shifted in the x-axis.
    y_shift : float
        The number of pixels that the array is shifted in the y-axis.
    mode : str, default = "constant"
        The padding mode of the translation. It must be one of the following.
        The implimentation detail is similar to Scipy's. See
        :py:func:`scipy.ndimage.shift` for more information.
    constant : float, default = np.nan
        If the `mode` is constant, the constant value used is this value.

    Returns
    -------
    translated : ndarray
        The translated array/image.

    """
    # Small conversions to make sure the inputs are proper.
    mode = str(mode).casefold()

    # We ensure that the array is 2D, or rather, image like.
    image_dimensions = 2
    if len(array.shape) != image_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Translating an array with shape {array.shape} via an"
                " image translation is not possible."
            ),
        )

    # We then apply the shift.
    shifted_array = scipy.ndimage.shift(
        array,
        (y_shift, x_shift),
        mode=mode,
        cval=constant,
    )
    return shifted_array


def rotate_2d(
    array: hint.ndarray,
    rotation: float,
    mode: str = "constant",
    constant: float = np.nan,
) -> hint.ndarray:
    """Rotate a 2D image array array.

    This function is a connivent wrapper around scipy's function.

    Parameters
    ----------
    array : ndarray
        The input array to be rotated.
    rotation : float
        The rotation angle, in radians.
    mode : str, default = "constant"
        The padding mode of the translation. It must be one of the following.
        The implementation detail is similar to Scipy's. See
        :py:func:`scipy.ndimage.shift` for more information.
    constant : float, default = np.nan
        If the `mode` is constant, the constant value used is this value.

    Returns
    -------
    rotated_array : ndarray
        The rotated array/image.

    """
    # Small conversions to make sure the inputs are proper.
    mode = str(mode).casefold()

    # We ensure that the array is 2D, or rather, image like.
    image_dimensions = 2
    if len(array.shape) != image_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Rotating an image array with shape {array.shape} via an"
                " image rotation is not possible."
            ),
        )

    # The scipy function takes the angle as degrees, so we need to convert.
    rotation_deg = (180 / np.pi) * rotation

    # We then apply the shift.
    rotated_array = scipy.ndimage.rotate(
        array,
        rotation_deg,
        mode=mode,
        cval=constant,
    )
    return rotated_array
