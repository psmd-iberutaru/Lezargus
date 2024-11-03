"""Array or image transformations, typically affine transformations.

The transform of images and arrays are important, and here we separate many
similar functions into this module.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import numpy as np
import scipy.ndimage

import lezargus
from lezargus.library import logging


def translate_2d(
    array: hint.NDArray,
    x_shift: float,
    y_shift: float,
    mode: str = "constant",
    constant: float = np.nan,
) -> hint.NDArray:
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
    array: hint.NDArray,
    rotation: float,
    mode: str = "constant",
    constant: float = np.nan,
) -> hint.NDArray:
    """Rotate a 2D image array.

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


def crop_2d(
    array: hint.NDArray,
    new_shape: tuple,
    location: str | tuple = "center",
    use_pillow: bool = False,
) -> hint.NDArray:
    """Crop a 2D image array.

    Parameters
    ----------
    array : ndarray
        The input array to be cropped.
    new_shape : tuple
        The new shape of the array after cropping.
    location : str | tuple, default = "center"
        The central location of the crop, provided as either a pixel coordinate
        or an instruction as follows:

        - center : The center of the array.
    use_pillow : bool, default = False
        If True, we use the PIL/Pillow module to determine the crop.

    Returns
    -------
    crop : ndarray
        The cropped array.

    """
    # Keeping.
    lezargus.library.wrapper.do_nothing(use_pillow)

    # Basic properties.
    current_shape = array.shape

    # We first define the location.
    if isinstance(location, str):
        location = location.casefold()
        if location == "center":
            center_location = current_shape[0] // 2, current_shape[1] // 2
        else:
            logging.error(
                error_type=logging.InputError,
                message=f"Location instruction {location} is not valid.",
            )
            return array
    else:
        center_location = location

    # Now we define the pixel locations for the crop.
    x_left = center_location[0] - int(np.floor(new_shape[0] / 2))
    x_right = center_location[0] + int(np.ceil(new_shape[0] / 2))
    y_bot = center_location[1] - int(np.floor(new_shape[1] / 2))
    y_top = center_location[1] + int(np.ceil(new_shape[1] / 2))
    # Returning the crop.
    crop = array[x_left:x_right, y_bot:y_top].copy()
    return crop


def scale_2d(
    array: hint.NDArray,
    new_shape: tuple,
) -> hint.NDArray:
    """Todo..."""
    return array, new_shape
