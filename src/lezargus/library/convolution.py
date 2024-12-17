"""Convolution functions and kernel producing functions.

Here, we group all convolution functions and kernel functions. A lot of the
convolution functions are brief wrappers around Astropy's convolution.
All three dimensionalities are covered.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import astropy.convolution
import astropy.modeling
import numpy as np

from lezargus.library import logging


def _check_array_dimensionality(array: hint.NDArray, dimensions: int) -> bool:
    """Check if the array has the expected number of dimensions.

    This function checks if the array has the correction number of
    dimensions. Of course, the expected dimensions are different so this
    function is more a wrapper around the logging message and it
    serves as a basic check.

    Parameters
    ----------
    array : ndarray
        The array that we are testing if it has the same number of
        dimensions.
    dimensions : int
        The number of expected dimensions the array should have.

    Returns
    -------
    valid_dimensionality : bool
        If True, the array has the expected dimensionality, as input.

    """
    # We just use Numpy shape and the like, type conversion.
    array = np.asarray(array)
    dimensions = int(dimensions)

    # Checking.
    valid_dimensionality = len(array.shape) == dimensions
    if not valid_dimensionality:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Input array has wrong dimensions, shape {array.shape};"
                f" expected dimensionality {dimensions}."
            ),
        )
    return valid_dimensionality


def _check_kernel_dimensionality(kernel: hint.NDArray, dimensions: int) -> bool:
    """Check if the kernel has the expected number of dimensions.

    Same function as :py:meth:`_check_array_dimensionality`, just different
    error message.

    Parameters
    ----------
    kernel : ndarray
        The kernel that we are testing if it has the same number of
        dimensions.
    dimensions : int
        The number of expected dimensions the kernel should have.

    Returns
    -------
    valid_dimensionality : bool
        If True, the kernel has the expected dimensionality, as input.

    """
    # We just use Numpy shape and the like, type conversion.
    kernel = np.asarray(kernel)
    dimensions = int(dimensions)

    # Checking.
    valid_dimensionality = len(kernel.shape) == dimensions
    if not valid_dimensionality:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Input kernel has wrong dimensions, shape {kernel.shape};"
                f" expected dimensionality {dimensions}."
            ),
        )
    return valid_dimensionality


def _check_array_kernel_variable_stack(
    array: hint.NDArray,
    kernel_stack: hint.NDArray,
    axis: int | tuple[int],
) -> bool:
    """Check if the kernel stack and array have the exact slice count.

    For variable kernels, we need to make sure that there are enough kernels
    in the kernel stack for each slice of the array which are being convolved.
    The axes which are variable (and not dynamic, i.e. changing with
    the convolution axis) are checked to be the same size in the array and
    the kernel stack.

    Parameters
    ----------
    array : ndarray
        The array which would be convolved and which we are checking is
        compatible with the kernel stack.
    kernel_stack : ndarray
        The kernel stack which holds all of the kernels that are being used.
        We are checking if it is compatible with the kernel stack.
    axis : int, tuple[int]
        Either a single axis index, or a tuple of axis indexes which the
        kernel is variable. The axis or axes should not be the same as the
        convolution axis or axes.

    Returns
    -------
    valid_stack : bool
        If True, the kernel stack has the correct amount of kernels for the
        array, for the axes which are variable.

    """
    # We repackage the axis, if it is a single value, it is the same as just
    # checking one so the code below can be the same.
    if isinstance(axis, int):
        check_axes = (axis,)
    elif isinstance(axis, list | tuple | np.ndarray):
        # It is already a tuple-like.
        check_axes = tuple(int(axisdex) for axisdex in axis)
    else:
        check_axes = axis
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Cannot parse axis input {axis} to a list of axis indexes to"
                " check."
            ),
        )

    # Just loop over each of the axes, checking if they have the same size.
    # We assume a good stack at first.
    valid_stack = True
    mismatched_axes = []
    for axisdex in check_axes:
        # We need to make sure that there is enough kernels in the stack for
        # each of the array slices.
        array_stack_count = array.shape[axisdex]
        kernel_stack_count = kernel_stack.shape[axisdex]
        if array_stack_count != kernel_stack_count:
            # The axes are mismatched, we record the mismatch here to report
            # later.
            valid_stack = False
            mismatched_axes.append(axisdex)

    # Is the stack still good?
    if not valid_stack:
        # Nope, we report the invalid stack. There is no one to one relation
        # for the array and kernel.
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Array shape {array.shape} incompatible with kernel stack"
                f" {kernel_stack.shape}. Mismatch in {mismatched_axes} indexed"
                " axes."
            ),
        )
    # All done.
    return valid_stack


def _static_astropy_convolve(
    array: hint.NDArray,
    kernel: hint.NDArray,
) -> hint.NDArray:
    """Use Astropy to convolve the array provided the kernel.

    The Astropy convolve function only can convolve up to 3D, and they
    determine it based on the array and kernel dimensionality. We attempt
    to do an FFT convolution, but, should it fail, we fall back to
    discrete convolution.

    Parameters
    ----------
    array : ndarray
        The array we are convolving by the kernel.
    kernel : ndarray
        The kernel we are using to convolve.

    Returns
    -------
    convolved : ndarray
        The result of the convolution.

    """
    # The array and kernel should have the same dimensionality, and it
    # cannot be more than 3. We base it on the array as that is the more
    # fundamental part.
    dimensionality = len(array.shape)
    _check_array_dimensionality(array=array, dimensions=dimensionality)
    _check_kernel_dimensionality(kernel=kernel, dimensions=dimensionality)

    # Checking if Astropy's convolve function can handle it.
    max_astropy_convolve_dimensionality = 3
    if dimensionality > max_astropy_convolve_dimensionality:
        logging.critical(
            critical_type=logging.NotSupportedError,
            message=(
                "Astropy convolve only supports up to 3D arrays, input"
                f" dimensionality is {dimensionality}."
            ),
        )

    # FFT convolution uses complex numbers. We want to keep the same
    # numerical precision as the input type. We can expand this to 192-bit
    # and 256-bit, but, it is likely not needed.
    if array.dtype.itemsize * 2 <= np.complex64(None).itemsize:
        complex_data_type = np.complex64
    elif array.dtype.itemsize * 2 <= np.complex128(None).itemsize:
        complex_data_type = np.complex128
    else:
        complex_data_type = complex

    # There are two ways that the convolution can happen, either via FFT
    # or via discrete convolution. It is always faster to do it via FFT
    # but we fall back to discrete convolution if we run out of memory.
    try:
        convolved = astropy.convolution.convolve_fft(
            array,
            kernel=kernel,
            boundary="wrap",
            complex_dtype=complex_data_type,
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
            allow_huge=True,
        )
    except MemoryError:
        # There is not enough memory for an FFT version, using discrete
        # instead.
        # We give some warning first.
        logging.warning(
            warning_type=logging.MemoryFullWarning,
            message=(
                "Attempting a FFT convolution of a spectra with shape"
                f" {array.shape} with kernel shape {kernel.shape} requires"
                " too much memory."
            ),
        )
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Discrete convolution attempted as an alternative to FFT"
                " convolution due to memory issues; expect long execution time."
            ),
        )
        # Discrete convolution.
        convolved = astropy.convolution.convolve(
            array,
            kernel=kernel,
            boundary="extend",
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
        )
    # All done.
    return convolved


def static_1d_with_1d(
    array: hint.NDArray,
    kernel: hint.NDArray,
) -> hint.NDArray:
    """Convolve a 1D array using a static 1D kernel.

    Parameters
    ----------
    array : ndarray
        The 1D array data which we will convolve.
    kernel : ndarray
        The 1D kernel that we are using to convolve.

    Returns
    -------
    convolved : ndarray
        The convolved 1D array data.

    """
    # We check that both the array and the kernel has the proper
    # dimensionality.
    array_dimensionality = 1
    kernel_dimensionality = 1
    _check_array_dimensionality(
        array=array,
        dimensions=array_dimensionality,
    )
    _check_kernel_dimensionality(
        kernel=kernel,
        dimensions=kernel_dimensionality,
    )

    # We do the convolution.
    convolved = _static_astropy_convolve(array=array, kernel=kernel)
    return convolved


def static_2d_with_2d(
    array: hint.NDArray,
    kernel: hint.NDArray,
) -> hint.NDArray:
    """Convolve a 2D array using a static 2D kernel.

    Parameters
    ----------
    array : ndarray
        The 2D array data which we will convolve.
    kernel : ndarray
        The 2D kernel that we are using to convolve.

    Returns
    -------
    convolved : ndarray
        The convolved 2D array data.

    """
    # We check that both the array and the kernel has the proper
    # dimensionality.
    array_dimensionality = 2
    kernel_dimensionality = 2
    _check_array_dimensionality(
        array=array,
        dimensions=array_dimensionality,
    )
    _check_kernel_dimensionality(
        kernel=kernel,
        dimensions=kernel_dimensionality,
    )

    # We do the convolution.
    convolved = _static_astropy_convolve(array=array, kernel=kernel)
    return convolved


def static_3d_with_1d_over_z(
    array: hint.NDArray,
    kernel: hint.NDArray,
) -> hint.NDArray:
    """Convolve a 3D array using a 1D kernel, over the z dimension.

    This convolution convolves 1D slices of the 3D array. The convolution
    itself then is a 1D array being convolved with a 1D kernel. We take slices
    of the last dimension (z), iterating over the 1st and 2nd dimension.

    Parameters
    ----------
    array : ndarray
        The 3D array data which we will convolve.
    kernel : ndarray
        The 1D kernel that we are using to convolve over the z axis of the
        array.

    Returns
    -------
    convolved : ndarray
        The convolved 3D array data.

    """
    # We check that both the array and the kernel has the proper
    # dimensionality.
    array_dimensionality = 3
    kernel_dimensionality = 1
    _check_array_dimensionality(
        array=array,
        dimensions=array_dimensionality,
    )
    _check_kernel_dimensionality(
        kernel=kernel,
        dimensions=kernel_dimensionality,
    )

    # Applying the convolution. This really is just a repeated process of
    # 1D convolutions across both other axes. We create the resulting
    # array and fill it in with the results of the convolutions.
    convolved = np.empty_like(array)
    for coldex in np.arange(array.shape[0]):
        for rowdex in np.arange(array.shape[1]):
            convolved[coldex, rowdex, :] = static_1d_with_1d(
                array=array[coldex, rowdex, :],
                kernel=kernel,
            )
    # All done.
    return convolved


def static_3d_with_2d_over_xy(
    array: hint.NDArray,
    kernel: hint.NDArray,
) -> hint.NDArray:
    """Convolve a 3D array using a 2D kernel, over the x-y plane.

    This convolution convolves 2D slices of the 3D array. The convolution
    itself then is a 2D array being convolved with a 3D kernel. A full
    3D array and 3D kernel convolution is not done here.

    Parameters
    ----------
    array : ndarray
        The 3D array data which we will convolve.
    kernel : ndarray
        The 2D kernel that we are using to convolve over the x-y plane
        of the array.

    Returns
    -------
    convolved : ndarray
        The convolved 3D array data.

    """
    # We check that both the array and the kernel has the proper
    # dimensionality.
    array_dimensionality = 3
    kernel_dimensionality = 2
    _check_array_dimensionality(
        array=array,
        dimensions=array_dimensionality,
    )
    _check_kernel_dimensionality(
        kernel=kernel,
        dimensions=kernel_dimensionality,
    )

    # Applying the convolution. This really is just a repeated process of
    # 2D convolutions across the x-y plane. We create the resulting
    # array and fill it in with the results of the convolutions.
    convolved = np.empty_like(array)
    for index in np.arange(array.shape[2]):
        convolved[:, :, index] = static_2d_with_2d(
            array=array[:, :, index],
            kernel=kernel,
        )
    # All done.
    return convolved


def variable_3d_with_2d_over_xy(
    array: hint.NDArray,
    kernel_stack: hint.NDArray,
) -> hint.NDArray:
    """Convolve a 3D array using a variable 2D kernel, over the x-y plane.

    Like py:func:`static_3d_with_2d_over_xy`, this convolution convolves
    2D slices of the 3D array. However, the kernel here is variable in
    in the z dimension.

    Parameters
    ----------
    array : ndarray
        The 3D array data which we will convolve.
    kernel_stack : ndarray
        The 2D kernel stack that we are using to convolve over the x-y
        plane of the array. Each slice of the stack should correspond to
        the kernel for the slice of the array.

    Returns
    -------
    convolved : ndarray
        The convolved 3D array data.

    """
    # It is best to work with arrays.
    kernel_stack = np.asarray(kernel_stack)

    # We check that both the array and the kernel has the proper
    # dimensionality. The kernel stack is extra by one because of the
    # stack axis.
    array_dimensionality = 3
    kernel_dimensionality = 3
    _check_array_dimensionality(array=array, dimensions=array_dimensionality)
    _check_kernel_dimensionality(
        kernel=kernel_stack,
        dimensions=kernel_dimensionality,
    )

    # We also check if the array and kernel stack is compatible for variable
    # convolution along the varying axis.
    varying_axis = 2
    is_valid_stack = _check_array_kernel_variable_stack(
        array=array,
        kernel_stack=kernel_stack,
        axis=varying_axis,
    )
    if not is_valid_stack:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Kernel stack mismatch,"
                f" {kernel_stack.shape[varying_axis]} kernels for"
                f" {array.shape[varying_axis]} array slices."
            ),
        )

    # Applying the convolution. This really is just a repeated process of
    # 2D convolutions across the x-y plane, except we also iterate the
    # kernel stack.
    convolved = np.empty_like(array)
    for index in np.arange(array.shape[2]):
        convolved[:, :, index] = static_2d_with_2d(
            array=array[:, :, index],
            kernel=kernel_stack[:, :, index],
        )
    # All done.
    return convolved


def kernel_1d_gaussian(
    shape: tuple | int,
    stddev: float,
) -> hint.NDArray:
    """Return a 1D Gaussian convolution kernel.

    We normalize the kernel via the amplitude of the Gaussian
    function as a whole for maximal precision: volume = 1. The `stddev` must
    be expressed in pixels.

    Parameters
    ----------
    shape : tuple | int
        The shape of the 1D kernel, in pixels. If a single value (i.e. a size
        value instead), we attempt convert it to a shape-like value.
    stddev : float
        The standard deviation of the Gaussian, in pixels.

    Returns
    -------
    gaussian_kernel : ndarray
        The discrete kernel array.

    """
    # We need to make sure we can handle odd inputs of the standard
    # deviation, just in case.
    if stddev < 0:
        logging.error(
            error_type=logging.InputError,
            message=f"Gaussian stddev {stddev}is negative, not physical.",
        )
    elif np.isclose(stddev, 0):
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                f"Gaussian stddev is {stddev}, about zero; kernel is basically"
                " a delta-function."
            ),
        )

    # We need to determine the shape. If it is a single value we attempt to
    # interpret it. Granted, we only need a size, but we keep a shape as the
    # input to align it better with the 2D kernel functions.
    if isinstance(shape, list | tuple) and len(shape) == 1:
        # All good.
        size = shape[0]
    elif isinstance(shape, int | np.number):
        size = shape
    else:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Kernel shape input {shape} type {type(shape)} is not a 1D"
                " array shape."
            ),
        )
        size = shape
    # Regardless, the center of the array is considered to be the center of
    # the Gaussian function.
    center = (size - 1) / 2
    # The actual input array to the Gaussian function.
    input_ = np.arange(size, dtype=int)

    # The normalization constant is really just the area of the Gaussian.
    norm_constant = 1 / (stddev * np.sqrt(2 * np.pi))

    # Deriving the kernel and computing it.
    gaussian1d = astropy.modeling.models.Gaussian1D(
        amplitude=norm_constant,
        mean=center,
        stddev=stddev,
    )
    gaussian_kernel = gaussian1d(input_)
    # All done.
    return gaussian_kernel


def kernel_1d_gaussian_resolution(
    shape: tuple | int,
    template_wavelength: hint.NDArray | float,
    base_resolution: float | None = None,
    target_resolution: float | None = None,
    base_resolving_power: float | None = None,
    target_resolving_power: float | None = None,
    reference_wavelength: float | None = None,
) -> hint.NDArray:
    """Gaussian 1D kernel adapted for resolution convolution conversions.

    This function is a wrapper around a normal 1D Gaussian kernel. Instead
    of specifying the standard deviation, we calculate the approximate
    required standard deviation needed to down-sample a base resolution to
    some target resolution. We accept both resolution values or resolving
    power values for the calculation; but we default to resolution based
    determination if possible.

    Parameters
    ----------
    shape : tuple | int
        The shape of the 1D kernel, in pixels. If a single value (i.e. a size
        value instead), we attempt convert it to a shape-like value.
    template_wavelength : ndarray or float
        An example wavelength array which this kernel will be applied to. This
        is required to convert the physical standard deviation value calculated
        from the resolution/resolving power to one of length in pixels/points.
        If an array, we try and compute the conversion factor. If a float,
        that is the conversion factor of wavelength per pixel.
    base_resolution : float, default = None
        The base resolution that we are converting from. Must be provided
        along with `target_resolution` for the resolution mode.
    target_resolution : float, default = None
        The target resolution we are converting to. Must be provided
        along with `base_resolution` for the resolution mode.
    base_resolving_power : float, default = None
        The base resolving power that we are converting from. Must be provided
        along with `target_resolving_power` and `reference_wavelength` for the
        resolving power mode.
    target_resolving_power : float, default = None
        The target resolving power that we are converting from. Must be
        provided along with `base_resolving_power` and `reference_wavelength`
        for the resolving power mode.
    reference_wavelength : float, default = None
        The reference wavelength used to convert from resolving power to
        resolution. Must be provided along with `base_resolving_power` and
        `target_resolving_power` for the resolving power mode.

    Returns
    -------
    resolution_kernel : ndarray
        The Gaussian kernel with the appropriate parameters to convert from
        the base resolution to the target resolution with a convolution.

    """
    # We support two different modes of computing the kernel. Toggle is based
    # on what parameters are provided. We switch here.
    resolution_mode = (
        base_resolution is not None and target_resolution is not None
    )
    resolving_mode = (
        base_resolving_power is not None
        and target_resolving_power is not None
        and reference_wavelength is not None
    )

    # Determining which, and based on which, we determine the determine the
    # standard deviation for the Gaussian. However, the standard deviation
    # value determined here is a physical length, not one in pixels/points.
    phys_fwhm = -1
    if resolution_mode and resolving_mode:
        # If we have both modes, the program cannot decide between both.
        # Though we default to resolution based modes, it is still problematic.
        logging.error(
            error_type=logging.InputError,
            message=(
                "Both resolution mode and resolving mode information was"
                " provided for kernel determination. Mode cannot be determined."
            ),
        )
        phys_fwhm = np.sqrt(target_resolution**2 - base_resolution**2)
    elif resolution_mode:
        # Resolution mode, we determine the standard deviation from the
        # provided resolutions.
        phys_fwhm = np.sqrt(target_resolution**2 - base_resolution**2)
    elif resolving_mode:
        # Resolving mode, we determine the standard deviation from the
        # provided resolving power and root wavelength.
        phys_fwhm = reference_wavelength * (
            (base_resolving_power**2 - target_resolving_power**2)
            / (base_resolving_power * target_resolving_power)
        )
        logging.warning(
            warning_type=logging.DeprecationWarning,
            message=(
                "Resolving power kernel generation should be replaced with"
                " resolutions computed from resolving power."
            ),
        )
    else:
        # No mode could be found usable. The inputs seem to be quite wrong.
        # This is equivalent to TypeError missing argument, hence a critical
        # failure.
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Kernel calculation mode could not be determined. Resolution"
                f" mode values: base, {base_resolution}; target:"
                f" {target_resolution}. Resolving mode values: base,"
                f" {base_resolving_power}; target, {target_resolving_power};"
                f" wavelength, {reference_wavelength}."
            ),
        )

    # Before we continue, we need to make sure that the FWHM is reasonable.
    # If it is not reasonable, we try and find potential problems to warn
    # about.
    if phys_fwhm <= 0:
        # If the resolutions are the same, it basically leads to a
        # delta function.
        if resolution_mode and np.isclose(target_resolution, base_resolution):
            logging.warning(
                warning_type=logging.AlgorithmWarning,
                message=(
                    f"Target resolution {target_resolution} and base resolution"
                    f" {base_resolution} is the same. Possible delta kernel."
                ),
            )
        # Similar, if the resolving powers are the same, it basically leads to a
        # delta function.
        if resolving_mode and np.isclose(
            target_resolving_power,
            base_resolving_power,
        ):
            logging.warning(
                warning_type=logging.AlgorithmWarning,
                message=(
                    f"Target resolving power {target_resolving_power} and base"
                    f" resolving power {base_resolving_power} is the same."
                    " Possible delta kernel."
                ),
            )

    # Converting to standard deviation.
    fwhm_std_const = 2 * np.sqrt(2 * np.log(2))
    phys_stddev = phys_fwhm / fwhm_std_const

    # We convert the physical standard deviation into a standard deviation of
    # pixels (or points in general). We assume a wavelength spacing
    # based on the average spacing of the provided wavelength.
    if isinstance(template_wavelength, float | int | np.number):
        convert_factor = template_wavelength
    else:
        convert_factor = np.nanmean(
            template_wavelength[1:] - template_wavelength[:-1],
        )
    # Converting
    stddev = phys_stddev / convert_factor

    # With the standard deviation known, we can compute the kernel using the
    # Gaussian kernel creator.
    resolution_kernel = kernel_1d_gaussian(shape=shape, stddev=stddev)
    # All done.
    return resolution_kernel


def kernel_2d_gaussian(
    shape: tuple,
    x_stddev: float,
    y_stddev: float,
    rotation: float,
) -> hint.NDArray:
    """Return a 2D Gaussian convolution kernel.

    We normalize the kernel via the amplitude of the Gaussian
    function as a whole for maximal precision: volume = 1. We require the
    input of the shape of the kernel to allow for `x_stddev` and `y_stddev`
    to be expressed in pixels to keep it general. By definition, the center
    of the Gaussian kernel is in the center of the array.

    Parameters
    ----------
    shape : tuple
        The shape of the 2D kernel, in pixels.
    x_stddev : float
        The standard deviation of the Gaussian in the x direction, in pixels.
    y_stddev : float
        The standard deviation of the Gaussian in the y direction, in pixels.
    rotation : float
        The rotation angle, increasing counterclockwise, in radians.

    Returns
    -------
    gaussian_kernel : ndarray
        The discrete kernel array.

    """
    # The center of the array given by the shape is defined as just the center
    # of it. However, we need to take into account off-by-one errors.
    try:
        nrow, ncol = shape
    except ValueError:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "The 2D kernel shape cannot be determined from input shape:"
                f" {shape}"
            ),
        )
    cen_row = (nrow - 1) / 2
    cen_col = (ncol - 1) / 2

    # The normalization constant is provided as volume itself.
    norm_constant = 1 / (2 * np.pi * x_stddev * y_stddev)

    # The mesh grid used to evaluate the Gaussian function to derive the kernel.
    xx, yy = np.meshgrid(np.arange(ncol, dtype=int), np.arange(nrow, dtype=int))

    # Deriving the kernel and computing it.
    gaussian2d = astropy.modeling.models.Gaussian2D(
        amplitude=norm_constant,
        x_mean=cen_col,
        y_mean=cen_row,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=rotation,
    )
    gaussian_kernel = gaussian2d(xx, yy)
    return gaussian_kernel
