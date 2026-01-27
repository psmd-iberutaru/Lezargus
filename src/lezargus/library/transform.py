"""Array or image transformations, typically affine; and computer vision tools.

The transform of images and arrays are important, and here we separate many
similar functions into this module. Other related functions are stored in
this module as well including computer vision tools like corner detection,
[[BLAW]], and [[BLAW]] among other things.
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
    order: int = 3,
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
    order : int, default = 3
        The order of the spline interpolation, default is 3. The order has
        to be in the range 0-5.
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
        order=order,
        mode=mode,
        cval=constant,
    )
    return rotated_array


def crop_2d(
    array: hint.NDArray,
    new_shape: tuple,
    location: str | tuple = "center",
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

    Returns
    -------
    crop : ndarray
        The cropped array.

    """
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
    # our points will be considered as inliers, we don't need to fiddle with
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


def corner_detection(
    array: hint.NDArray,
    max_corners: int = -1,
    quality_level: float | None = None,
    minimum_distance: float = 1,
    use_harris: bool = False,
) -> list[tuple]:
    """Detect corners in an gray image array.

    This function is a half-wrapper of the Shi-Tomasi corner detection
    algorithm per OpenCV's implementation. (Though, a flag exists to use
    the Harris 1988 corner detection algorithm instead as per OpenCV's
    function.) See `cv2.goodFeaturesToTrack()` for more information.

    Parameters
    ----------
    array : NDArray
        The image array which we will be using to detect the corners.
        This should be either an 8-bit int or a 32-bit float array.
    max_corners : int | None, default = None
        The maximum number of corners to find. If provided only the "best"
        corners, based on the detection metric, up to the maximum will be
        provided. If 0 or a negative value (default), all valid
        corners will be returned instead as per OpenCV.
    quality_level : float | None, default = None
        The minimum quality level required for the corners. This value is a
        factor and establishes the quality floor of detected corners as a
        ratio to the highest quality corner found. If None, the default
        value will be exceedingly low ensuring `max_corners` find the
        required corners. This should be set if `max_corners` is not set.
    minimum_distance : float, default = 1
        The minimum Euclidean distance between the detected corners, in
        units of pixels. By default, we use 1, to prevent the same corner
        being detected multiple times.
    use_harris : bool, default = False
        If True, we use the Harris method instead of the Shi-Tomasi method
        per OpenCV. Default is False, that is, to use the method as per
        usual.

    Returns
    -------
    corners : list
        A list of the (x, y) coordinate tuple pairs of the corners found.
        The tuple is ranked in order of the quality of the corners.

    """
    # Defaults.
    # A negative value expressly asks for all corners found. However, a
    # current bug in OpenCV or its documentation requires the flag to be
    # exactly 0.
    max_corners = int(max(0, max_corners))
    # The quality level cannot be 0 but we can make it really close to it.
    # Using the 32-bit float just to be quick.
    quality_level = float(2e-38 if quality_level is None else quality_level)
    # And the minimum distance, should be greater than 1.
    minimum_distance = max(1, minimum_distance)

    # Some warnings about if too many corners are to be made...
    low_quality_level = 1e-20
    if (max_corners <= 0) and (quality_level <= low_quality_level):
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                f"No `max_corners` and a low quality level {quality_level} will"
                " likely provide useless corners."
            ),
        )

    # OpenCV requires the data type of the array to be specific; either
    # 8-bit integers or 32-bit floats. Though customary, it is not required
    # for the float array to be normalized so we do not do it here.
    array_dtype = array.dtype
    if np.isdtype(array_dtype, np.dtype(np.int8)) or np.isdtype(
        array_dtype,
        np.dtype(np.float32),
    ):
        # All good.
        valid_array = array
    # It is not one of the valid arrays... We can see if we can convert it
    # to one.
    # Starting with integers first as floats are more "lenient".
    elif np.can_cast(array_dtype, np.dtype(np.int8)):
        valid_array = np.asarray(array, dtype=np.dtype(np.int8))
    elif np.can_cast(array_dtype, np.dtype(np.float32)):
        valid_array = np.asarray(array, dtype=np.dtype(np.float32))
    else:
        # We try our best with 32 bit floats... But it is useful to let the
        # user know of the bad types.
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                f"Casting {array_dtype} to 32-bit float for OpenCV corner"
                " detection."
            ),
        )
        try:
            valid_array = np.array(array, dtype=np.dtype(np.float32))
        except TypeError:
            # Even the problematic cast to float 32 is wrong. This will more
            # than likely be a problem.
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Array with type {array_dtype} cannot be converted to"
                    " the expected 8-bit int or 32-bit float as expected by"
                    " OpenCV."
                ),
            )
            valid_array = array

    # Finding the corners using OpenCV.
    # For some reason, the corners are embedded in a too-high a dimension pair
    # group.
    raw_corner_output = cv2.goodFeaturesToTrack(
        image=valid_array,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=minimum_distance,
        useHarrisDetector=use_harris,
    )
    corner_output = np.squeeze(raw_corner_output)

    # We repackage the output per the documentation.
    corners = [(int(xdex), int(ydex)) for (xdex, ydex) in corner_output]
    return corners


def corner_detection_subpixel_refinement(
    array: hint.NDArray,
    initial_corners: list,
    search_radius: int,
    iterations: int = 1000,
) -> list[tuple]:
    """Refine the initial detected corners with sub-pixel accuracy.

    This function does not detect corners themselves, but refines already
    made detections with subpixel accuracy. This function is a half-wrapper
    of the OpenCV `cv2.cornerSubPix()` function. See their documentation
    for more information.

    Parameters
    ----------
    array : NDAarray
        The image array which we will be using to refine the corner detection.
        This should be a 32-bit float array, or convertible to 32-bit float.
    initial_corners : list
        The initial corners which we will be refining with this method,
        provided as a list of (x, y) tuple points.
    search_radius : int
        The pixel search radius around which the function will look for a
        better corner. This is technically a half-width that defines the
        search bounding square centered on the initial corner.
    iterations : int, default = 1000
        The number of iterations of the corner refinement method. This is
        passed straight to OpenCV.

    Returns
    -------
    refined_corners : list
        A list of the (x, y) refined coordinate tuple pairs of the corners
        found after the subpixel refinement. The order of the points is the
        same as the order as the input initial corners.

    """
    # We need to make sure that the array data is in the right data type,
    # as OpenCV requires (it seems) that both the data and the initial points
    # are 32-bit float.
    array_dtype = array.dtype
    if np.isdtype(array_dtype, np.dtype(np.float32)):
        # All good.
        valid_array = array
    elif np.can_cast(array_dtype, np.dtype(np.float32)):
        # The data can be cast to the right type so it is okay.
        valid_array = np.asarray(array, dtype=np.dtype(np.float32))
    else:
        # We try our best with 32-bit floats... But it is useful to let the
        # user know of the bad types.
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                f"Casting {array_dtype} to 32-bit float for OpenCV corner"
                " detection refinement."
            ),
        )
        try:
            valid_array = np.array(array, dtype=np.dtype(np.float32))
        except TypeError:
            # Even the problematic cast to float 32 is wrong. This will more
            # than likely be a problem.
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Array with type {array_dtype} cannot be converted to"
                    " the expected 32-bit float as expected by OpenCV."
                ),
            )
            valid_array = array

    # We need to make sure the initial points are also of the right type.
    # Using much less care as they are just initial conditions for the points.
    try:
        using_initial_corners = np.array(initial_corners, dtype=np.float32)
    except ValueError:
        # The corners cannot be cast to float for some reason?
        using_initial_corners = initial_corners
        logging.error(
            error_type=logging.InputError,
            message=(
                "The initial corners provided cannot be cast to a single"
                " 32-bit float array, as expected by OpenCV."
            ),
        )

    # Defining the criteria of the corner detection optimization. We stop
    # when either maximum iterations have been met, or epsilon change is
    # small enough, so we use both flags for the criteria type.
    criteria_type = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
    max_iterations = iterations
    max_epsilon = 0.0001
    criteria = (criteria_type, max_iterations, max_epsilon)

    # The bounding box window, the real search window in OpenCV is (2w, 2w).
    # The dead zone is the region where it will not search in, though it is
    # rare for us to use this region so we define no such size.
    half_width_window = (search_radius, search_radius)
    half_width_dead_zone = (-1, -1)

    # Finding the refined points. This function overwrites the initial corner
    # variable as a way to output the data too, which is not normal Python.
    # We make sure it does it just here.
    __initial_corners = np.array(using_initial_corners, copy=True)
    raw_refined_output = cv2.cornerSubPix(
        valid_array,
        __initial_corners,
        half_width_window,
        half_width_dead_zone,
        criteria,
    )
    raw_refined_output = np.squeeze(raw_refined_output)

    # We repackage the output per the documentation.
    refined_corners = [
        (float(xdex), float(ydex)) for (xdex, ydex) in raw_refined_output
    ]
    # All done.
    return refined_corners
