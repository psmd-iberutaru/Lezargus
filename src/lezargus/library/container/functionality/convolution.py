"""Convolution functions specifically tailored to Lezargus containers.

We seperate the logic for containers into functions which make it a little
easier to understand. Moreover, these functions can also be used separately.
The logic is similar to Numpy's functions like py:func:`numpy.mean` and
`numpy.ndarray.mean`; and other modules in
:py:mod:`lezargus.library.container.functionality`.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import numpy as np

import lezargus
from lezargus.library import logging


def convolve_spectrum_by_spectral_kernel(
    spectrum: hint.LezargusSpectrum,
    kernel: hint.NDArray | None = None,
    kernel_stack: hint.NDArray | None = None,
    kernel_function: hint.Callable | None = None,
) -> hint.LezargusSpectrum:
    """Convolve the spectrum with a spectral kernel.

    We compute the convolution and return a near copy of the spectrum after
    convolution. The wavelength is not affected.

    As spectrum are 1D, there is no dimension to have a variable kernel,
    as variable kernels need a non-convolution axis to vary on. There can
    only be a static or dynamic kernel.

    Parameters
    ----------
    spectrum : LezargusCube
        The spectrum we are convolving.
    kernel : ndarray, default = None
        A static 1D spectral kernel. If provided, we use this static kernel
        to convolve the spectrum by. Exclusive with other kernel options.
    kernel_stack : ndarray, default = None
        A variable 1D spectral kernel stack. If provided, we use the
        variable kernel stack to convolve the spectrum by. Exclusive with
        other kernel options.
    kernel_function : Callable, default = None
        A dynamic 1D kernel function. If provided, we use the dynamic kernel
        function to convolve the spectrum by. Exclusive with other kernel
        options.

    Returns
    -------
    convolved_spectrum : ndarray
        A near copy of the spectrum after convolution.

    """
    # Determine the kernel used, and the convolution function being used.
    is_static = kernel is not None
    is_variable = kernel_stack is not None
    is_dynamic = kernel_function is not None
    if sum((is_static, is_variable, is_dynamic)) != 1:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Only one kernel type allowed: static, {is_static};"
                f" variable, {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We assume that variances add; thus uncertainties add in quadrature.
    variance = spectrum.uncertainty**2

    # We convolve based on the method of convolution as specified.
    if is_static:
        # Static convolution.
        convolved_data = lezargus.library.convolution.static_1d_with_1d(
            array=spectrum.data,
            kernel=kernel,
        )
        convolved_variance = lezargus.library.convolution.static_1d_with_1d(
            array=variance,
            kernel=kernel,
        )
    elif is_variable:
        logging.warning(
            warning_type=logging.AlgorithmWarning,
            message=(
                "Variable kernel convolution with a Spectrum undefined, no"
                " non-convolution axis. Assuming static kernel."
            ),
        )
        # We still try static convolution. Will likely fail.
        convolved_data = lezargus.library.convolution.static_1d_with_1d(
            array=spectrum.data,
            kernel=kernel,
        )
        convolved_variance = lezargus.library.convolution.static_1d_with_1d(
            array=variance,
            kernel=kernel,
        )
    elif is_dynamic:
        # Dynamic convolution.
        convolved_data = None
        convolved_variance = None
        logging.critical(
            critical_type=logging.ToDoError,
            message="No cube dynamic convolution.",
        )
    else:
        # One of the above convolutions should have triggered.
        convolved_data = spectrum.data
        convolved_variance = variance
        logging.error(
            error_type=logging.LogicFlowError,
            message=(
                f"Unknown convolution mode: static, {is_static}; variable,"
                f" {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We also propagate the convolution of the mask and the flags where
    # needed.
    logging.error(
        error_type=logging.ToDoError,
        message="Propagation of mask and flags via convolution is not done.",
    )
    convolved_mask = spectrum.mask
    convolved_flags = spectrum.flags

    # Converting back to uncertainty.
    convolved_uncertainty = np.sqrt(convolved_variance)

    # From the above information, we construct the new spectrum.
    spectrum_class = type(spectrum)
    convolved_spectrum = spectrum_class(
        wavelength=spectrum.wavelength,
        data=convolved_data,
        uncertainty=convolved_uncertainty,
        wavelength_unit=spectrum.wavelength_unit,
        data_unit=spectrum.data_unit,
        mask=convolved_mask,
        flags=convolved_flags,
        header=spectrum.header,
    )

    # All done.
    return convolved_spectrum


def convolve_cube_by_spectral_kernel(
    cube: hint.LezargusCube,
    kernel: hint.NDArray | None = None,
    kernel_stack: hint.NDArray | None = None,
    kernel_function: hint.Callable | None = None,
) -> hint.LezargusCube:
    """Convolve the cube by a spectral kernel convolving spectra slices.

    Convolving a spectral cube can either be done one of two ways;
    convolving by image slices or convolving by spectral slices. We here
    convolve by spectral slices.

    Parameters
    ----------
    cube : LezargusCube
        The cube we are convolving.
    kernel : ndarray, default = None
        A static 1D spectral kernel. If provided, we use this static kernel
        to convolve the cube by. Exclusive with other kernel options.
    kernel_stack : ndarray, default = None
        A variable 1D spectral kernel stack. If provided, we use the
        variable kernel stack to convolve the cube by. Exclusive with
        other kernel options.
    kernel_function : Callable, default = None
        A dynamic 1D kernel function. If provided, we use the dynamic kernel
        function to convolve the cube by. Exclusive with other kernel
        options.

    Returns
    -------
    convolved_cube : ndarray
        A near copy of the data cube after convolution.

    """
    # Determine the kernel used, and the convolution function being used.
    is_static = kernel is not None
    is_variable = kernel_stack is not None
    is_dynamic = kernel_function is not None
    if sum((is_static, is_variable, is_dynamic)) != 1:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Only one kernel type allowed: static, {is_static};"
                f" variable, {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We assume that variances add; thus uncertainties add in quadrature.
    variance = cube.uncertainty**2

    # We convolve based on the method of convolution as specified.
    if is_static:
        # Static convolution.
        convolved_data = lezargus.library.convolution.static_3d_with_1d_over_z(
            array=cube.data,
            kernel=kernel,
        )
        convolved_variance = (
            lezargus.library.convolution.static_3d_with_1d_over_z(
                array=variance,
                kernel=kernel,
            )
        )
    elif is_variable:
        # Variable convolution.
        convolved_data = None
        convolved_variance = None
        logging.critical(
            critical_type=logging.ToDoError,
            message="No cube by spectra variable convolution.",
        )
    elif is_dynamic:
        # Dynamic convolution.
        convolved_data = None
        convolved_variance = None
        logging.critical(
            critical_type=logging.ToDoError,
            message="No cube by spectra dynamic convolution.",
        )
    else:
        # One of the above convolutions should have triggered.
        convolved_data = cube.data
        convolved_variance = variance
        logging.error(
            error_type=logging.LogicFlowError,
            message=(
                f"Unknown convolution mode: static, {is_static}; variable,"
                f" {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We also propagate the convolution of the mask and the flags where
    # needed.
    logging.error(
        error_type=logging.ToDoError,
        message="Propagation of mask and flags via convolution is not done.",
    )
    convolved_mask = cube.mask
    convolved_flags = cube.flags

    # Converting back to uncertainty.
    convolved_uncertainty = np.sqrt(convolved_variance)

    # From the above information, we construct the new spectra.
    cube_class = type(cube)
    convolved_cube = cube_class(
        wavelength=cube.wavelength,
        data=convolved_data,
        uncertainty=convolved_uncertainty,
        wavelength_unit=cube.wavelength_unit,
        data_unit=cube.data_unit,
        spectral_scale=cube.spectral_scale,
        pixel_scale=cube.pixel_scale,
        slice_scale=cube.slice_scale,
        mask=convolved_mask,
        flags=convolved_flags,
        header=cube.header,
    )

    # All done.
    return convolved_cube


def convolve_cube_by_image_kernel(
    cube: hint.LezargusCube,
    kernel: hint.NDArray | None = None,
    kernel_stack: hint.NDArray | None = None,
    kernel_function: hint.Callable | None = None,
) -> hint.LezargusCube:
    """Convolve the cube by an image kernel convolving image slices.

    Convolving a spectral cube can either be done one of two ways;
    convolving by image slices or convolving by spectral slices. We here
    convolve by image slices.

    Parameters
    ----------
    cube : LezargusCube
        The cube we are convolving.
    kernel : ndarray, default = None
        A static 2D image kernel. If provided, we use this static kernel
        to convolve the cube by. Exclusive with other kernel options.
    kernel_stack : ndarray, default = None
        A variable 2D image kernel stack. If provided, we use the variable
        kernel stack to convolve the cube by. Exclusive with other kernel
        options.
    kernel_function : Callable, default = None
        A dynamic 2D kernel function. If provided, we use the dynamic
        kernel function to convolve the cube by. Exclusive with other
        kernel options.

    Returns
    -------
    convolved_cube : ndarray
        A near copy of the data cube after convolution.

    """
    # Determine the kernel used, and the convolution function being used.
    is_static = kernel is not None
    is_variable = kernel_stack is not None
    is_dynamic = kernel_function is not None
    if sum((is_static, is_variable, is_dynamic)) != 1:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Only one kernel type allowed: static, {is_static};"
                f" variable, {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We assume that variances add; thus uncertainties add in quadrature.
    variance = cube.uncertainty**2

    # We convolve based on the method of convolution as specified.
    if is_static:
        # Static convolution.
        convolved_data = lezargus.library.convolution.static_3d_with_2d_over_xy(
            array=cube.data,
            kernel=kernel,
        )
        convolved_variance = (
            lezargus.library.convolution.static_3d_with_2d_over_xy(
                array=variance,
                kernel=kernel,
            )
        )
    elif is_variable:
        # Variable convolution.
        convolved_data = (
            lezargus.library.convolution.variable_3d_with_2d_over_xy(
                array=cube.data,
                kernel_stack=kernel_stack,
            )
        )
        convolved_variance = (
            lezargus.library.convolution.variable_3d_with_2d_over_xy(
                array=variance,
                kernel_stack=kernel_stack,
            )
        )
    elif is_dynamic:
        # Dynamic convolution.
        convolved_data = None
        convolved_variance = None
        logging.critical(
            critical_type=logging.ToDoError,
            message="No cube by image dynamic convolution.",
        )
    else:
        # One of the above convolutions should have triggered.
        convolved_data = cube.data
        convolved_variance = variance
        logging.error(
            error_type=logging.LogicFlowError,
            message=(
                f"Unknown convolution mode: static, {is_static}; variable,"
                f" {is_variable}; dynamic, {is_dynamic}."
            ),
        )

    # We also propagate the convolution of the mask and the flags where
    # needed.
    logging.error(
        error_type=logging.ToDoError,
        message="Propagation of mask and flags via convolution is not done.",
    )
    convolved_mask = cube.mask
    convolved_flags = cube.flags

    # Converting back to uncertainty.
    convolved_uncertainty = np.sqrt(convolved_variance)

    # From the above information, we construct the new spectra.
    cube_class = type(cube)
    convolved_cube = cube_class(
        wavelength=cube.wavelength,
        data=convolved_data,
        uncertainty=convolved_uncertainty,
        wavelength_unit=cube.wavelength_unit,
        data_unit=cube.data_unit,
        spectral_scale=cube.spectral_scale,
        pixel_scale=cube.pixel_scale,
        slice_scale=cube.slice_scale,
        mask=convolved_mask,
        flags=convolved_flags,
        header=cube.header,
    )

    # All done.
    return convolved_cube
