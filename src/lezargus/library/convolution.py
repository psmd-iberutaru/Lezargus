"""Convolution functions and kernel producing functions.

Here, we group all convolution functions and kernel functions. A lot of the
convolution functions are brief wrappers around Astropy's convolution.
All three dimensionalities are covered.
"""

import astropy.convolution
import astropy.modeling
import numpy as np

from lezargus.library import hint
from lezargus.library import logging


def convolve_1d_array_by_1d_kernel(
    array: hint.ndarray,
    kernel: hint.ndarray,
) -> hint.ndarray:
    """Convolve a 1D array using a 1D kernel.

    Parameters
    ----------
    array : ndarray
        The 1D array data which we will convolve.
    kernel : ndarray
        The 1D kernel that we are using to convolve.

    Returns
    -------
    convolved_array : ndarray
        The convolved 1D array data.
    """
    # We need to ensure that the convolution array and kernel are the proper
    # dimensions.
    array_dimensions = 1
    kernel_dimensions = 1
    if len(array.shape) != array_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input array is not actually a 1D array, shape is"
                f" {array.shape}. Applying convolution with a {kernel.shape}"
                " kernel shape may fail."
            ),
        )
    if len(kernel.shape) != kernel_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input kernel is not actually a 1D array, shape is"
                f" {kernel.shape}. Applying convolution with a {array.shape}"
                " array may fail."
            ),
        )

    # We want to keep the same numerical precision, or rather, as close as
    # we can to the original data type. We can expand this to 192-bit and
    # 256-bit, but, it is likely not needed.
    if array.dtype.itemsize * 2 <= np.complex64(None).itemsize:
        complex_data_type = np.complex64
    elif array.dtype.itemsize * 2 <= np.complex128(None).itemsize:
        complex_data_type = np.complex128
    else:
        complex_data_type = complex

    # We don't really expect a 1D spectra convolution to run into memory
    # issues, but we still need to anticipate it. We try FFT convolution first,
    # then we go to a discrete convolution if it fails.
    try:
        convolved_array = astropy.convolution.convolve_fft(
            array,
            kernel=kernel,
            boundary="fill",
            fill_value=np.nanmedian(array),
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
                "Discrete convolution will be attempted as an alternative to"
                " the FFT convolution due to memory issues."
            ),
        )
        # Discrete convolution.
        convolved_array = astropy.convolution.convolve(
            array,
            kernel=kernel,
            boundary="extend",
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
        )
    # All done.
    return convolved_array


def convolve_2d_array_by_2d_kernel(
    array: hint.ndarray,
    kernel: hint.ndarray,
) -> hint.ndarray:
    """Convolve a 2D array using a 2D kernel.

    Parameters
    ----------
    array : ndarray
        The 2D array data which we will convolve.
    kernel : ndarray
        The 2D kernel that we are using to convolve.

    Returns
    -------
    convolved_array : ndarray
        The convolved 2D array data.
    """
    # We need to ensure that the convolution array and kernel are the proper
    # dimensions.
    array_dimensions = 2
    kernel_dimensions = 2
    if len(array.shape) != array_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input array is not actually a 2D array, shape is"
                f" {array.shape}. Applying convolution with a {kernel.shape}"
                " kernel shape may fail."
            ),
        )
    if len(kernel.shape) != kernel_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input kernel is not actually a 2D array, shape is"
                f" {kernel.shape}. Applying convolution with a {array.shape}"
                " array may fail."
            ),
        )

    # We want to keep the same numerical precision, or rather, as close as
    # we can to the original data type. We can expand this to 192-bit and
    # 256-bit, but, it is likely not needed.
    if array.dtype.itemsize * 2 <= np.complex64(None).itemsize:
        complex_data_type = np.complex64
    elif array.dtype.itemsize * 2 <= np.complex128(None).itemsize:
        complex_data_type = np.complex128
    else:
        complex_data_type = complex

    # We attempt to do the 2D convolution using FFT. However, FFT can be
    # memory intensive so we default back to the standard discrete version
    # if there is not enough memory.  For the fill value, the most common
    # value is likely to be sky noise so we just pad it with sky noise.
    try:
        convolved_array = astropy.convolution.convolve_fft(
            array,
            kernel=kernel,
            boundary="fill",
            fill_value=np.nanmedian(array),
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
                "Attempting a FFT convolution of an image with shape"
                f" {array.shape} with kernel shape {kernel.shape} requires too"
                " much memory."
            ),
        )
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Discrete convolution will be attempted as an alternative to"
                " the FFT convolution due to memory issues."
            ),
        )
        # Discrete convolution.
        convolved_array = astropy.convolution.convolve(
            array,
            kernel=kernel,
            boundary="extend",
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=True,
        )
    # All done,
    return convolved_array


def convolve_3d_by_1d_kernel(
    array: hint.ndarray,
    kernel: hint.ndarray,
) -> hint.ndarray:
    """Convolve a 3D array using a 1D kernel, looping 2 dimensions.

    This convolution convolves 1D slices of the 3D array. The convolution
    itself then is a 1D array being convolved with a 1D kernel. We take slices
    of the last dimension, iterating over the 1st and 2nd dimension. A full
    3D array and 3D kernel convolution is not done here.

    Parameters
    ----------
    array : ndarray
        The 3D array data which we will convolve.
    kernel : ndarray
        The 1D kernel that we are using to convolve.

    Returns
    -------
    convolved_array : ndarray
        The convolved 3D array data.
    """
    # We need to ensure that the convolution array and kernel are the proper
    # dimensions.
    array_dimensions = 3
    kernel_dimensions = 1
    if len(array.shape) != array_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input array is not actually a 3D array, shape is"
                f" {array.shape}. Applying slice convolution with a"
                f" {kernel.shape} kernel shape may fail."
            ),
        )
    if len(kernel.shape) != kernel_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input kernel is not actually a 1D array, shape is"
                f" {kernel.shape}. Applying slice convolution with a"
                f" {array.shape} array may fail."
            ),
        )

    # Applying the convolution. For the fill value, the most common value is
    # likely to be sky noise so we just pad it with sky noise. Moreover, some
    # of these cubes can be rather large. However, sometimes this process can
    # be very memory intensive so we need to be able to fallback to a backup.
    convolved_array = np.zeros_like(array)

    # This really is just a repeated process of 2D convolutions.
    for coldex in np.arange(array.shape[0]):
        for rowdex in np.arange(array.shape[1]):
            convolved_array[coldex, rowdex, :] = convolve_1d_array_by_1d_kernel(
                array=array[coldex, rowdex, :],
                kernel=kernel,
            )
    # All done.
    return convolved_array


def convolve_3d_by_2d_kernel(
    array: hint.ndarray,
    kernel: hint.ndarray,
) -> hint.ndarray:
    """Convolve a 3D array using a 2D kernel, looping over the 3rd dimension.

    This convolution convolves 2D slices of the 3D array. The convolution
    itself then is a 2D array being convolved with a 3D kernel. A full
    3D array and 3D kernel convolution is not done here.

    Parameters
    ----------
    array : ndarray
        The 3D array data which we will convolve.
    kernel : ndarray
        The 2D kernel that we are using to convolve.

    Returns
    -------
    convolved_array : ndarray
        The convolved 3D array data.
    """
    # We need to ensure that the convolution array and kernel are the proper
    # dimensions.
    array_dimensions = 3
    kernel_dimensions = 2
    if len(array.shape) != array_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input array is not actually a 3D array, shape is"
                f" {array.shape}. Applying slice convolution with a"
                f" {kernel.shape} kernel shape may fail."
            ),
        )
    if len(kernel.shape) != kernel_dimensions:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "The input kernel is not actually a 2D array, shape is"
                f" {kernel.shape}. Applying slice convolution with a"
                f" {array.shape} array may fail."
            ),
        )

    # Applying the convolution. For the fill value, the most common value is
    # likely to be sky noise so we just pad it with sky noise. Moreover, some
    # of these cubes can be rather large. However, sometimes this process can
    # be very memory intensive so we need to be able to fallback to a backup.
    convolved_array = np.zeros_like(array)

    # This really is just a repeated process of 2D convolutions.
    for index in np.arange(array.shape[2]):
        convolved_array[:, :, index] = convolve_2d_array_by_2d_kernel(
            array=array[:, :, index],
            kernel=kernel,
        )
    # All done.
    return convolved_array


def kernel_1d_gaussian(
    shape: tuple | int,
    stddev: float,
) -> hint.ndarray:
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
    template_wavelength: hint.ndarray | float,
    base_resolution: float | None = None,
    target_resolution: float | None = None,
    base_resolving_power: float | None = None,
    target_resolving_power: float | None = None,
    reference_wavelength: float | None = None,
) -> hint.ndarray:
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
) -> hint.ndarray:
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
