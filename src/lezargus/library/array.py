"""Collection of array or image manipulation functions.

If there are any functions which are done on arrays (or anything that is
just an array under the hood), we usually group it here. Moreover, functions
which would otherwise operate on images are also placed here. As images are
just arrays under the hood (and to avoid conflict with
/lezargus/container/image.py), image manipulation functions are kept here too.

Note that all of these functions follow the axes convention of indexing being
(x, y, lambda). If a cube is not of this shape, then it will likely return
erroneous results, but, the functions themselves cannot detect this.
"""

import numpy as np
import scipy.ndimage

from lezargus.library import hint
from lezargus.library import logging


def translate_image_array(
    input_array: hint.ndarray,
    x_shift: float,
    y_shift: float,
) -> hint.ndarray:
    """Translate a 2D image array array.

    This function is a convenient wrapper around scipy's function. The array is
    padded with NaNs so any data outside the original array after translation
    is null.

    Parameters
    ----------
    input_array : ndarray
        The input array to be translated.
    x_shift : float
        The number of pixels that the array is shifted in the x-axis.
    y_shift : float
        The number of pixels that the array is shifted in the y-axis.

    Returns
    -------
    shifted_array : ndarray
        The shifted array/image.

    """
    # We ensure that the array is 2D, or rather, image like.
    image_dimensions = 2
    if len(input_array.shape) != image_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Translating an array with shape {input_array.shape} via an"
                " image translation is not possible."
            ),
        )

    # We then apply the shift.
    shifted_array = scipy.ndimage.shift(
        input_array,
        (y_shift, x_shift),
        order=3,
        mode="constant",
        cval=np.nan,
    )
    return shifted_array


def rotate_image_array(
    input_array: hint.ndarray,
    rotation: float,
) -> hint.ndarray:
    """Rotate a 2D image array array.

    This function is a connivent wrapper around scipy's function. The array is
    padded with NaNs so any data outside the original array after rotation
    is null.

    Parameters
    ----------
    input_array : ndarray
        The input array to be rotated.
    rotation : float
        The rotation angle, in radians.

    Returns
    -------
    rotated_array : ndarray
        The rotated array/image.

    """
    # We ensure that the array is 2D, or rather, image like.
    image_dimensions = 2
    if len(input_array.shape) != image_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Rotating an image array with shape {input_array.shape} via an"
                " image rotation is not possible."
            ),
        )

    # The scipy function takes the angle as degrees, so we need to convert.
    rotation_deg = (180 / np.pi) * rotation

    # We then apply the shift.
    rotated_array = scipy.ndimage.rotate(
        input_array,
        rotation_deg,
        order=3,
        mode="constant",
        cval=np.nan,
    )
    return rotated_array


def bin_image_array(
    image: hint.ndarray,
    x_bin: int,
    y_bin: int,
    mode: str = "add",
) -> hint.ndarray:
    """Bin an image by using integer super pixels.

    A lot of inspiration for this function is from here:
    https://scipython.com/blog/binning-a-2d-array-in-numpy/

    Parameters
    ----------
    image : ndarray
        The input image/array to binned.
    x_bin : int
        The number of pixels in the x-direction to bin over per super pixel.
    y_bin : int
        The number of pixels in the y-direction to bin over per super pixel.
    mode : string, default = "add"
        The mode to combine the data.

            - `add` : Add the pixels together.
            - `mean` : Use the mean of the pixels.

    Returns
    -------
    binned_image : ndarray
        The image/array after binning.

    """
    # We need to check if the shape is compatible with the binning count.
    image_dimensions = 2
    if len(image.shape) != image_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=f"Binning an image array with shape {image.shape}.",
        )
    if image.shape[0] % x_bin != 0:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The x-axis dimension has length {image.shape[0]}, it is not"
                f" divisible by the bin pixel count {x_bin}."
            ),
        )
    if image.shape[1] % y_bin != 0:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The y-axis dimension has length {image.shape[1]}, it is not"
                f" divisible by the bin pixel count {y_bin}."
            ),
        )

    # We attempt to bin via reshaping and applying the function over the
    # provided bin. We group the binned pixels into dimensional extensions.
    new_shape = (image.shape[0] // x_bin, x_bin, image.shape[1] // y_bin, y_bin)
    new_image = image.reshape(new_shape)

    # We apply the combining function here based on the input.
    mode = mode.casefold()
    if mode == "add":
        binned_image = new_image.sum(axis=(1, 3))
    elif mode == "mean":
        binned_image = new_image.mean(axis=(1, 3))
    else:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The combining mode {mode} is not one of the supported modes:"
                " add, mean."
            ),
        )
    # All done.
    return binned_image


def bin_cube_array_spatially(
    cube: hint.ndarray,
    x_bin: int,
    y_bin: int,
    mode: str = "add",
) -> hint.ndarray:
    """Bin a cube spatially into super pixels.

    We only bin the cube in the spatial directions, the spectral direction is
    not touched.

    Parameters
    ----------
    cube : ndarray
        The data cube to binned.
    x_bin : int
        The number of pixels in the x-direction to bin over per super pixel.
    y_bin : int
        The number of pixels in the y-direction to bin over per super pixel.
    mode : string, default = "add"
        The mode to combine the data.

            - `add` : Add the pixels together.
            - `mean` : Use the mean of the pixels.

    Returns
    -------
    binned_image : ndarray
        The data cube after binning.

    """
    # We need to check if the shape is compatible with the binning count.\
    cube_dimensions = 3
    if len(cube.shape) != cube_dimensions:
        logging.error(
            error_type=logging.InputError,
            message=f"Binning a cube array with shape {cube.shape}.",
        )
    if cube.shape[0] % x_bin != 0:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The x-axis dimension has length {cube.shape[0]}, it is not"
                f" divisible by the bin pixel count {x_bin}."
            ),
        )
    if cube.shape[1] % y_bin != 0:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The y-axis dimension has length {cube.shape[1]}, it is not"
                f" divisible by the bin pixel count {y_bin}."
            ),
        )

    # We attempt to bin via reshaping and applying the function over the
    # provided bin. We group the binned pixels into dimensional extensions.
    new_shape = (
        cube.shape[0] // x_bin,
        x_bin,
        cube.shape[1] // y_bin,
        y_bin,
        cube.shape[2],
    )
    new_cube = cube.reshape(new_shape)

    # We apply the combining function here based on the input.
    mode = mode.casefold()
    if mode == "add":
        binned_cube = new_cube.sum(axis=(1, 3))
    elif mode == "mean":
        binned_cube = new_cube.mean(axis=(1, 3))
    else:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The combining mode {mode} is not one of the supported modes:"
                " add, mean."
            ),
        )
    # All done.
    return binned_cube
