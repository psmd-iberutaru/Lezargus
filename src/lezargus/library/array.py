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

import astropy.convolution
import numpy as np
import scipy.ndimage

from lezargus.library import hint
from lezargus.library import logging


def clean_finite_arrays(*arrays: hint.ndarray) -> tuple[hint.ndarray]:
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
        compatible, compatible_array = verify_shape_compatibility(
            reference_array=reference_array,
            test_array=arraydex,
            return_broadcast=True,
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


def verify_shape_compatibility(
    reference_array: hint.ndarray,
    test_array: hint.ndarray,
    return_broadcast: bool = False,
) -> bool | tuple[bool, hint.ndarray]:
    """Verify if a test array is compatible with the reference array.

    This function serves to see if two arrays are compatible in shape. If
    the "test" array is just a single number, we allow it to broadcast, and
    return it if needed.

    Parameters
    ----------
    reference_array : ndarray
        The reference array which we are testing against.
    test_array : ndarray
        The test array that we are testing to.
    return_broadcast : bool, default = False
        If True, we return a compatible array built from the test array
        information.

    Returns
    -------
    verify : bool
        The verification.
    broadcast : ndarray, optional
        The broadcast, we only return this if `return_broadcast` was provided.
        If verify is False, then this is None.
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
    if return_broadcast:
        return verify, broadcast
    # Otherwise, we just return the verification status only.
    return verify


def translate_image_array(
    input_array: hint.ndarray,
    x_shift: float,
    y_shift: float,
) -> hint.ndarray:
    """Translate a 2D image array array.

    This function is a convient wrapper around scipy's function. The array is
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

            * `add` : Add the pixels together.
            * `mean` : Use the mean of the pixels.

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
        logging.error(
            error_type=logging.InputError,
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

            * `add` : Add the pixels together.
            * `mean` : Use the mean of the pixels.

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
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The combining mode {mode} is not one of the supported modes:"
                " add, mean."
            ),
        )
    # All done.
    return binned_cube


def convolve_cube_by_image_array(
    cube: hint.ndarray,
    kernel: hint.ndarray,
) -> hint.ndarray:
    """Convolve the image slices of a 3D cube with a 2D image.

    We loop over and convolve image slices of the cube with the provided
    kernel; we do not try and do an entire 3D convolution of the cube due to
    memory limitations.

    Parameters
    ----------
    cube : ndarray
        The data cube from which we will convolve.
    kernel : ndarray
        The image kernel we are using to convolve.

    Returns
    -------
    convolved_cube : ndarray
        The cube, with the image slices convolved by the provided kernel.
    """
    # We need to ensure that the cube is actually a cube.
    cube_dimensions = 3
    if len(cube.shape) != cube_dimensions:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "The input cube is not actually a cube, shape is {cube.shape}."
                " Applying convolution across image slices may fail."
            ),
        )

    # We want to keep the same numerical precision, or rather, as close as
    # we can to the original data type. We can expand this to 192-bit and
    # 256-bit, but, it is likely not needed.
    if cube.dtype.itemsize * 2 <= np.complex64(None).itemsize:
        complex_data_type = np.complex64
    elif cube.dtype.itemsize * 2 <= np.complex128(None).itemsize:
        complex_data_type = np.complex128
    else:
        complex_data_type = complex

    # Applying the convolution. For the fill value, the most common value is
    # likely to be sky noise so we just pad it with sky noise. Moreover, some
    # of these cubes can be rather large. However, sometimes this process can
    # be very memory intensive so we need to be able to fallback to a backup.
    convolved_cube = np.zeros_like(cube)
    try:
        # This really is just a repeated process of 2D convolutions.
        for index in np.arange(cube.shape[2]):
            convolved_cube[:, :, index] = astropy.convolution.convolve_fft(
                cube[:, :, index],
                kernel,
                boundary="fill",
                fill_value=np.nanmedian(cube),
                complex_dtype=complex_data_type,
                nan_treatment="interpolate",
                normalize_kernel=True,
                preserve_nan=True,
                allow_huge=True,
            )
    except MemoryError:
        logging.warning(
            warning_type=logging.MemoryFullWarning,
            message=(
                "Attempting a layered FFT convolution of a cube with shape"
                f" {cube.shape} with kernel shape {kernel.shape} requires too"
                " much memory."
            ),
        )
        # We use the alternative discrete convolution.
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Discrete convolution will be attempted as an alternative to"
                " the FFT convolution due to memory issues."
            ),
        )
        # Again, we try a repeated convolution but instead using the discrete
        # version.
        # An extended boundary condition is likely more representative of
        # sky noise then median.
        convolved_cube[:, :, index] = astropy.convolution.convolve(
            cube[:, :, index],
            kernel,
            boundary="extend",
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
        )
    # All done.
    return convolved_cube
