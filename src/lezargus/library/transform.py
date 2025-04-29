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

import cv2
import numpy as np
import scipy.ndimage

import lezargus
from lezargus.library import logging


def translate_2d(
    array: hint.NDArray,
    x_shift: float,
    y_shift: float,
    order: int = 3,
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
    order : int
        The spline order for the interpolation of the translation function.
    mode : str, default = "constant"
        The padding mode of the translation. It must be one of the following.
        The implementation detail is similar to Scipy's. See
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
        order=order,
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


def crop_3d(
    array: hint.NDArray,
    new_shape: tuple,
    location: str | tuple = "center",
    use_pillow: bool = False,
) -> hint.NDArray:
    """Crop a 3D image array.

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
            center_location = (
                current_shape[0] // 2,
                current_shape[1] // 2,
                current_shape[2] // 2,
            )
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
    z_back = center_location[2] - int(np.floor(new_shape[2] / 2))
    z_front = center_location[2] + int(np.ceil(new_shape[2] / 2))
    # Returning the crop.
    crop = array[x_left:x_right, y_bot:y_top, z_back:z_front].copy()
    return crop


def affine_transform(
    array: hint.NDArray,
    matrix: hint.NDArray,
    offset: hint.NDArray | None = None,
    constant: float | tuple = np.nan,
) -> hint.NDArray:
    """Execute an affine transformation on an array.

    This function only handles images.

    Parameters
    ----------
    array : ndarray
        The input array to be transformed.
    matrix : ndarray
        The transformation matrix. It may be homogenous, and if so,
        any input offset is ignored.
    offset : ndarray
        The translation offset of the affine transformation, specified if a
        homogenous matrix is not provided.
    constant : float | tuple, default = np.nan
        If the `mode` is constant, the constant value used is this value.
        Because we use OpenCV in the backend, a tuple representing a
        OpenCV Scalar may be provided.

    Returns
    -------
    transformed_array : ndarray
        The affine transformed array/image.

    """
    # We just use OpenCV's implementation.

    # Default for offset, if None.
    offset = np.zeros((2,)) if offset is None else offset

    # The matrix is required to be a 2x3 augmented matrix. We need to figure
    # it out from the provided matrix and offset.
    warp_matrix = np.zeros((2, 2))
    offset_vector = np.zeros((2,))
    if matrix.shape == (3, 3):
        # This matrix is homogenous.
        warp_matrix = matrix[0:2, 0:2]
        offset_vector = matrix[0:2, 2]
    elif matrix.shape == (2, 3):
        # The matrix is augmented already, we compute the warp and offset
        # even though it is the same.
        warp_matrix = matrix[0:2, 0:2]
        offset_vector = matrix[0:2, 2]
    elif matrix.shape == (2, 2):
        # The matrix is not augmented but is just a normal transformation
        # matrix.
        warp_matrix = matrix
        offset_vector = offset
    else:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Transformation matrix has shape {matrix.shape}, cannot create"
                " an affine matrix from it."
            ),
        )

    # Computing the augmented matrix for the transformation.
    augmented_matrix = np.insert(warp_matrix, 2, offset_vector, axis=1)

    # The border constant.
    border_constant = (0, 0, 0, 0)
    if isinstance(constant, tuple | list):
        # The border constant is likely a OpenCV scaler value.
        open_cv_scalar_length = 4
        if len(constant) == open_cv_scalar_length:
            border_constant = tuple(constant)
        else:
            logging.error(
                error_type=logging.InputError,
                message=(
                    "The border constant OpenCV Scaler is not a 4-element"
                    f" tuple: {constant}"
                ),
            )
    else:
        # It is likely a single value at this point.
        border_constant = (constant, constant, constant, constant)

    # OpenCV, for the shape parameter, uses a (width, height) convention,
    # while Numpy uses a (height, width) convention. We just need to adapt for
    # it.
    opencv_dsize = (array.shape[1], array.shape[0])

    # Transforming.
    transformed_array = cv2.warpAffine(
        src=array,
        M=augmented_matrix,
        dsize=opencv_dsize,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_constant,
    )

    return transformed_array


def calculate_affine_matrix(
    in_points: hint.NDArray,
    out_points: hint.NDArray,
) -> hint.NDArray:
    """Calculate the homogeneous affine transformation matrix from points.

    Provided a set of input and output point coordinates, and assuming
    an affine transformation between them, we calculate the optimal
    affine transformation as defined by a homogeneous affine transformation
    matrix. Generally, more than three pairs of points are provided and so
    we just find the best fit.

    Parameters
    ----------
    in_points : NDArray
        The set of input points, as an NxD array, for N number of points of
        D dimensions. Basically, the points transforming from to the output.
        Input and output should be parallel.
    out_points : NDArray
        The set of output points, as an NxD array, for N number of points of
        D dimensions. Basically, the points after the transform, from the input.
        Input and output should be parallel.

    Returns
    -------
    homogeneous_matrix : NDArray
        The best fitting homogeneous affine transformation matrix.

    """
    # Arranging the points as needed.
    in_points = np.array(in_points, dtype=float)
    out_points = np.array(out_points, dtype=float)

    # Determining the method.

    # Determining the registration. We use OpenCV here and given that most of
    # out points will be considered as inliers, we don't need to fiddle with
    # the RANSAC criterion.
    # We don't need the information about the inliers and outliers.
    augmented_matrix, __ = cv2.estimateAffine2D(
        from_=in_points,
        to=out_points,
        method=cv2.LMEDS,
    )

    # Standard affine transformation matrices don't store the translation
    # along with it, but we can do that using homogeneous matrixes, so we
    # make one from solution.
    homogeneous_matrix = np.insert(augmented_matrix, 2, [0, 0, 1], axis=0)

    # All done.
    return homogeneous_matrix
