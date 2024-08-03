"""Functions to properly broadcast one Lezargus container into another.

Sometimes operations are needed to be performed between two dimensions of
data structures. We have functions here which serve to convert from one
structure to another based on some broadcasting pattern. We properly handle
the internal conversions (such as the flags, mask, wavelength, etc) as well
based on the input template structure broadcasting to.
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


def broadcast_spectrum_to_cube(
    input_spectrum: hint.LezargusSpectrum,
    shape: tuple,
    location: tuple | hint.NDArray | str,
    fill_value: float = 0,
    fill_uncertainty: float = 0,
) -> hint.LezargusCube:
    """Make a LezargusCube from a LezargusSpectrum.

    A LezargusCube is made from a LezargusSpectrum, with its overall shape
    being defined. The location of where to broadcast the spectrum to the
    cube is also custom as well.

    Parameters
    ----------
    input_spectrum : LezargusSpectrum
        The input spectrum which will be broadcasted to fit the input template
        cube.
    shape : tuple
        The defined shape of the new cube. Either a two element tuple defining
        the spatial axes or a full three element tuple.
    location : tuple | ndarray | str
        The spatial location of where the spectrum is broadcast too. A
        single location is specified by a two element tuple.
        If a 2D array, all parts where True the spectrum is applied. If
        a string, instead we use the following instructions:

            - "center" : The spectrum is broadcast at the center, or close to
              it for the case of even edge shapes.
            - "full" : The spectrum is broadcast across the entire spatial
              area.

    fill_value : float, default = 0
        For the cube where there the spectrum is not being broadcast (i.e.
        outside the specified locations), we fill it with this data value.
    fill_uncertainty : float, default = 0
        Similar to py:param:`fill_value`, but for the uncertainty part of the
        cube.

    Returns
    -------
    broadcast_cube : LezargusCube
        The LezargusCube after the spectrum was uniformly broadcast spatially.
        Any header information came from first the spectrum then the cube.

    """
    # Ensure the input spectrum is a spectrum.
    if not isinstance(
        input_spectrum,
        lezargus.library.container.LezargusSpectrum,
    ):
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Input spectrum is type {type(input_spectrum)}, not a"
                " LezargusSpectrum."
            ),
        )

    # The provided cube shape must be compatible with the spectrum.
    image_spatial_shape = 2
    cube_spatial_shape = 3
    if len(shape) == image_spatial_shape:
        # It is a two element tuple defining the spatial shape.
        spatial_shape = shape
    elif len(shape) == cube_spatial_shape:
        # This is the full shape, we check that the wavelength axis is
        # the same shape.
        in_wave_len = shape[2]
        if in_wave_len != len(input_spectrum.wavelength):
            logging.error(
                error_type=logging.InputError,
                message=(
                    f"Shape {shape} specifies wavelength length {in_wave_len};"
                    " input spectrum wavelength length"
                    f" {len(input_spectrum.wavelength)}"
                ),
            )
        spatial_shape = (shape[0], shape[1])
    else:
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Input shape {shape} cannot be parsed to a cube specification."
            ),
        )
        spatial_shape = shape
    # We need the true shape of the cube do define the cube later.
    cube_shape = (*spatial_shape, len(input_spectrum.wavelength))

    # From the location, we derive a 2D spatial map. The map is how we
    # determine how to propagate the cube.
    spatial_map = None
    # The instructions can be converted to either a tuple location, or an
    # array.
    location = location.casefold() if isinstance(location, str) else location
    if location == "full":
        spatial_map = np.ones(spatial_shape, dtype=bool)
    elif location == "center":
        location = (shape[0] // 2, shape[1] // 2)
    # A tuple location can then be converted to a boolean array.
    point_pair_length = 2
    if (
        isinstance(location, tuple | list | np.ndarray)
        and len(location) == point_pair_length
    ):
        # A valid tuple location.
        spatial_map = np.zeros(spatial_shape, dtype=bool)
        spatial_map[*location] = True
    # If the location is already an array, then we just use it as the spatial
    # map.
    if isinstance(location, np.ndarray):
        spatial_map = np.asarray(location, dtype=bool)
    # A final check to make sure the map derived is compatible with the
    # cube.
    spatial_map_shape = None if spatial_map is None else spatial_map.shape
    if spatial_map_shape != spatial_shape:
        logging.error(
            error_type=logging.InputError,
            message=(
                "Spatial map derived from location has shape"
                f" {spatial_map_shape}, not compatible with cube {cube_shape}"
            ),
        )

    # We don't want to lose data resolution.
    input_dtype = input_spectrum.data.dtype

    # With the spatial map, and the spectrum, we can compute the data and
    # uncertainty cube broadcasts. We propagate the flags and masks as well.
    data_cube = np.full(cube_shape, fill_value=fill_value, dtype=input_dtype)
    uncertainty_cube = np.full(
        cube_shape,
        fill_value=fill_uncertainty,
        dtype=input_dtype,
    )
    mask_cube = np.zeros(cube_shape, dtype=bool)
    flags_cube = np.zeros(cube_shape, dtype=bool)
    # Applying the spectrum to where the spatial map specifies.
    data_cube[spatial_map, :] = input_spectrum.data
    uncertainty_cube[spatial_map, :] = input_spectrum.uncertainty
    mask_cube[spatial_map, :] = input_spectrum.mask
    flags_cube[spatial_map, :] = input_spectrum.flags

    # With the new broadcasted data, we can derive the new cube.
    broadcast_cube = lezargus.library.container.LezargusCube(
        wavelength=input_spectrum.wavelength,
        data=data_cube,
        uncertainty=uncertainty_cube,
        wavelength_unit=input_spectrum.wavelength_unit,
        data_unit=input_spectrum.data_unit,
        spectral_scale=input_spectrum.spectral_scale,
        pixel_scale=input_spectrum.pixel_scale,
        slice_scale=input_spectrum.slice_scale,
        mask=mask_cube,
        flags=flags_cube,
        header=input_spectrum.header,
    )
    # All done.
    return broadcast_cube
