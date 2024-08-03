"""Transformation functions specifically tailored to Lezargus containers.

We implement geometric transformations (typically affine-like transformations)
and their tailored implementation for Lezargus containers. Transformations
are special in that the specific implementation is very container specific
so care is needed in picking the right function.

We separate the logic for containers into functions which make it a little
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


def transform_shear_cube_spectral(
    cube: hint.LezargusCube,
    x_shifts: hint.NDArray,
    y_shifts: hint.NDArray,
    mode: str = "constant",
    constant: float = np.nan,
) -> hint.LezargusCube:
    """Apply a shear transformation along a cube's spectral axis.

    We translate the spatial slices of a cube, shearing along the cube's
    spectral axis. The other two (spatial) axes are not sheared across and
    remain non-transformed, just the data is translated; the shear is parallel
    to the spatial axes.

    Parameters
    ----------
    cube : LezargusCube
        The cube which we are going to apply a shear transformation along the
        spectral axis.
    x_shifts : ndarray
        The amount of shift in the x-axis of the spatial axes, in pixels.
        The length of this array must match the cube's spectral axis.
    y_shifts : ndarray
        The amount of shift in the y-axis of the spatial axes, in pixels.
        The length of this array must match the cube's spectral axis.
    mode : str, default = "constant"
        The padding mode of the shear translations. See
        :py:func:`lezargus.library.transform.translate_2d` for the avaliable
        options.
    constant : float, default = np.nan
        If the `mode` is constant, the constant value used is this value.

    Returns
    -------
    sheared_cube : LezargusCube
        The cube after the shear transformation was applied as instructed.

    """
    # We check that we have a Lezargus cube.
    if not isinstance(cube, lezargus.library.container.LezargusCube):
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Input cube is {type(cube)}, not a LezargusCube, undefined"
                " shear transformation."
            ),
        )
    # We check that there are enough shifts for each layer of the cube.
    wavelength_axis_length = len(cube.wavelength)
    if (
        len(x_shifts) != wavelength_axis_length
        or len(y_shifts) != wavelength_axis_length
    ):
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Not enough x_shifts {len(x_shifts)} or y_shifts"
                f" {len(y_shifts)} for cube wavelength axis size"
                f" {wavelength_axis_length}."
            ),
        )
    # Ensuring arrays, as we use indexing.
    x_shifts = np.asarray(x_shifts)
    y_shifts = np.asarray(y_shifts)

    # Storing the results.
    sheared_data = np.zeros_like(cube.data)
    sheared_uncertainty = np.zeros_like(cube.uncertainty)
    sheared_mask = np.zeros_like(cube.mask, dtype=bool)
    sheared_flags = np.zeros_like(cube.flags, dtype=int)
    # We can now begin to apply the transformation layer by layer.
    for index in range(wavelength_axis_length):
        # We translate the data and uncertainty.
        sheared_data[:, :, index] = lezargus.library.transform.translate_2d(
            array=cube.data[:, :, index],
            x_shift=x_shifts[index],
            y_shift=y_shifts[index],
            mode=mode,
            constant=constant,
        )
        sheared_uncertainty[:, :, index] = (
            lezargus.library.transform.translate_2d(
                array=cube.uncertainty[:, :, index],
                x_shift=x_shifts[index],
                y_shift=y_shifts[index],
                mode=mode,
                constant=constant,
            )
        )
        # Applying the translation on the mask and flags. We assume more
        # integer translations for this, combining the result of both
        # translations.
        # Mask translation is not done...
        # Flag translation is not done...
        sheared_mask[:, :, index] = cube.mask[:, :, index]
        sheared_flags[:, :, index] = cube.flags[:, :, index]

    # The handling of the masks and flags need to be done.
    logging.error(
        error_type=logging.ToDoError,
        message="Refraction handling of masks and flags need to be handled.",
    )

    # We have the translations applied, we can now reassemble the cube.
    # We allow for some subclassing, just in case.
    cube_type = type(cube)
    sheared_cube = cube_type(
        wavelength=cube.wavelength,
        data=sheared_data,
        uncertainty=sheared_uncertainty,
        wavelength_unit=cube.wavelength_unit,
        data_unit=cube.data_unit,
        pixel_scale=cube.pixel_scale,
        slice_scale=cube.slice_scale,
        mask=sheared_mask,
        flags=sheared_flags,
        header=cube.header,
    )
    # All done.
    return sheared_cube
