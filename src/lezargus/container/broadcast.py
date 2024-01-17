"""Functions to properly broadcast one Lezargus container into another.

Sometimes operations are needed to be performed between two dimensions of
data structures. We have functions here which serve to convert from one
structure to another based on some broadcasting pattern. We properly handle
the internal conversions (such as the flags, mask, wavelength, etc) as well
based on the input template structure broadcasting to.
"""

# This is a last resort solution to fixing the recursive import of the
# type hints here.
from __future__ import annotations

import numpy as np

import lezargus
from lezargus.library import hint
from lezargus.library import logging


def broadcast_spectra_to_cube_uniform(
    input_spectra: hint.LezargusSpectra,
    template_cube: hint.LezargusCube,
    wavelength_mode: str = "error",
) -> hint.LezargusCube:
    """Make a LezargusCube from a LezargusSpectra via uniform broadcasting.

    We make a LezargusCube, from a provided template, using uniform
    broadcasting. Uniform broadcasting is where all spectral slices, within
    the cube, are all the same for a uniform spatial distribution of the
    spectra, which is in this case the provided input spectra.

    In the case of both the input spectra and provided template cube having
    different wavelength arrays, we follow the provided mode to handle the
    different cases. The input template cube only provides the array shapes and
    the wavelength axis (dependant on the mode); the rest comes from the
    input spectra.

    Parameters
    ----------
    input_spectra : LezargusSpectra
        The input spectra which will be broadcasted to fit the input template
        cube.
    template_cube : LezargusCube
        The template cube which will serve as a template to determine the
        dimensional properties of the resulting broadcasting cube.
    wavelength_mode : str, default = "error"
        The mode to handle possible wavelength array conflicts between the
        spectra and the cube. The available options are:

            - "spectra" : Prefer the spectra's wavelength array; the cube's
            wavelength is ignored.
            - "cube" : Prefer the cube's wavelength array; the spectra is
            interpolated to align to the new wavelength.
            - "error" : We log an error. We still attempt to figure it out,
            defaulting to the spectra's wavelength array.


    Returns
    -------
    broadcast_cube : LezargusCube
        The LezargusCube after the spectra was uniformly broadcast spatially.
        Any header information came from first the spectra then the cube.
    """
    # First thing is first, type check the input.
    if not (
        isinstance(input_spectra, lezargus.container.LezargusSpectra)
        and isinstance(template_cube, lezargus.container.LezargusCube)
    ):
        # The objects are not the proper type so broadcasting them might go
        # really wrong.
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The input spectra type {type(input_spectra)} and template"
                f" cube type {type(template_cube)} are not instances of the"
                " expected LezargusSpectra and LezargusCube types"
                " respectively. Broadcasting may fail."
            ),
        )

    # Now, we need to determine the definitive wavelength array based on the
    # provided wavelength mode. However, the two wavelength units might be
    # different, this is problem that we ought to warn.
    if input_spectra.wavelength_unit != template_cube.wavelength_unit:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "The input spectra wavelength unit is"
                f" {input_spectra.wavelength_unit}, different from the template"
                f" cube wavelength unit {template_cube.wavelength_unit}"
            ),
        )
    # Regardless of the unit situation, we try our best to determine the
    # the preferred wavelength.
    wavelength_mode = wavelength_mode.casefold()
    if wavelength_mode == "spectra":
        # We rely on the spectra's wavelength.
        broadcast_wavelength = input_spectra.wavelength
        broadcast_wavelength_unit = input_spectra.wavelength_unit
    elif wavelength_mode == "cube":
        # We rely on the cube's wavelength.
        broadcast_wavelength = template_cube.wavelength
        broadcast_wavelength_unit = template_cube.wavelength_unit
    elif wavelength_mode == "error":
        # If the wavelengths differ, we raise an error on their mismatch.
        if not np.all(
            np.isclose(input_spectra.wavelength, template_cube.wavelength),
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input spectra and template cube wavelength arrays do not"
                    " match; wavelength mode is `error`; returning None."
                ),
            )
        # Regardless if the error was logged or not, we use the input spectra
        # as the broadcast.
        broadcast_wavelength = input_spectra.wavelength
        broadcast_wavelength_unit = input_spectra.wavelength_unit
    else:
        # The input parameter is not a given parameter.
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The input wavelength mode {wavelength_mode} is not a"
                " supported option."
            ),
        )

    # Finally, we determine the appropriate data based on the spectra and
    # the preferred wavelength array.
    (
        interpolated_data,
        interpolated_uncertainty,
        interpolated_mask,
        interpolated_flags,
    ) = input_spectra.interpolate(
        wavelength=broadcast_wavelength,
        skip_mask=False,
        skip_flags=False,
    )
    # The data unit for the data and the like.
    broadcast_data_unit = input_spectra.data_unit

    # Now, we assemble the cube. We only need the spatial coverage of the cube
    # and broadcast our 1D spectra to the spatial dimensions. We do not
    # really care for the wavelength shape of the cube.
    x_dim, y_dim, __ = template_cube.data.shape
    wave_dim = broadcast_wavelength.shape[0]
    # Building the cubes.
    broadcast_data = np.broadcast_to(
        interpolated_data,
        shape=(x_dim, y_dim, wave_dim),
    )
    broadcast_uncertainty = np.broadcast_to(
        interpolated_uncertainty,
        shape=(x_dim, y_dim, wave_dim),
    )
    broadcast_mask = np.broadcast_to(
        interpolated_mask,
        shape=(x_dim, y_dim, wave_dim),
    )
    broadcast_flags = np.broadcast_to(
        interpolated_flags,
        shape=(x_dim, y_dim, wave_dim),
    )

    # Finally, we reconstruct the cube. We work on copies of the headers
    # just in case.
    spectra_header = input_spectra.header.copy()
    cube_header = template_cube.header.copy()
    broadcast_header = cube_header.update(spectra_header)

    # Building the new broadcasted cube. We use the template's cube's class
    # just in case it has been subclassed or something.
    template_cube_class = type(template_cube)
    broadcast_cube = template_cube_class(
        wavelength=broadcast_wavelength,
        data=broadcast_data,
        uncertainty=broadcast_uncertainty,
        wavelength_unit=broadcast_wavelength_unit,
        data_unit=broadcast_data_unit,
        mask=broadcast_mask,
        flags=broadcast_flags,
        header=broadcast_header,
    )
    # All done.
    return broadcast_cube


def broadcast_spectra_to_cube_center(
    input_spectra: hint.LezargusSpectra,
    template_cube: hint.LezargusCube,
    wavelength_mode: str = "error",
    allow_even_center: bool = True,
) -> hint.LezargusCube:
    """Make a LezargusCube from a LezargusSpectra via center broadcasting.

    We make a LezargusCube, from a provided template, using center
    broadcasting. Center broadcasting is the provided spectral slice is
    centered in the cube; all other values are zero (or the blank equivalent
    for masks and flags). If any side of an image slice of the template cube
    is even, we can still try to place the image in the center, biasing it
    towards the lower value corner.

    In the case of both the input spectra and provided template cube having
    different wavelength arrays, we follow the provided mode to handle the
    different cases. The input template cube only provides the array shapes and
    the wavelength axis (dependant on the mode); the rest comes from the
    input spectra.

    Parameters
    ----------
    input_spectra : LezargusSpectra
        The input spectra which will be broadcasted to fit the input template
        cube.
    template_cube : LezargusCube
        The template cube which will serve as a template to determine the
        dimensional properties of the resulting broadcasting cube.
    wavelength_mode : str, default = "error"
        The mode to handle possible wavelength array conflicts between the
        spectra and the cube. The available options are:

            - "spectra" : Prefer the spectra's wavelength array; the cube's
            wavelength is ignored.
            - "cube" : Prefer the cube's wavelength array; the spectra is
            interpolated to align to the new wavelength.
            - "error" : We log an error and return None.
    allow_even_center : bool, default = True
        If True, and if any axis of an image slice is even, a warning is
        logged and the spectra is put it as close to the center as possible.
        If False, instead, an exception is raised.

    Returns
    -------
    broadcast_cube : LezargusCube
        The LezargusCube after the spectra was center broadcast spatially.
        Any header information came from first the spectra then the cube.
    """
    # First thing is first, type check the input.
    if not (
        isinstance(input_spectra, lezargus.container.LezargusSpectra)
        and isinstance(template_cube, lezargus.container.LezargusCube)
    ):
        # The objects are not the proper type so broadcasting them might go
        # really wrong.
        logging.error(
            error_type=logging.InputError,
            message=(
                f"The input spectra type {type(input_spectra)} and template"
                f" cube type {type(template_cube)} are not instances of the"
                " expected LezargusSpectra and LezargusCube types"
                " respectively. Broadcasting may fail."
            ),
        )

    # Now, we need to determine the definitive wavelength array based on the
    # provided wavelength mode. However, the two wavelength units might be
    # different, this is problem that we ought to warn.
    if input_spectra.wavelength_unit != template_cube.wavelength_unit:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                "The input spectra wavelength unit is"
                f" {input_spectra.wavelength_unit}, different from the template"
                f" cube wavelength unit {template_cube.wavelength_unit}"
            ),
        )
    # Regardless of the unit situation, we try our best to determine the
    # the preferred wavelength.
    wavelength_mode = wavelength_mode.casefold()
    if wavelength_mode == "spectra":
        # We rely on the spectra's wavelength.
        broadcast_wavelength = input_spectra.wavelength
        broadcast_wavelength_unit = input_spectra.wavelength_unit
    elif wavelength_mode == "cube":
        # We rely on the cube's wavelength.
        broadcast_wavelength = template_cube.wavelength
        broadcast_wavelength_unit = template_cube.wavelength_unit
    elif wavelength_mode == "error":
        # If the wavelengths differ, we raise an error on their mismatch.
        if not np.all(
            np.isclose(input_spectra.wavelength, template_cube.wavelength),
        ):
            logging.error(
                error_type=logging.InputError,
                message=(
                    "Input spectra and template cube wavelength arrays do not"
                    " match; wavelength mode is `error`; returning None."
                ),
            )
            broadcast_wavelength = None
            broadcast_wavelength_unit = None
            return None
        # Otherwise, if they do match, we can continue with the broadcasting.
        broadcast_wavelength = input_spectra.wavelength
        broadcast_wavelength_unit = input_spectra.wavelength_unit
    else:
        # The input parameter is not a given parameter.
        logging.critical(
            critical_type=logging.InputError,
            message=(
                f"The input wavelength mode {wavelength_mode} is not a"
                " supported option."
            ),
        )

    # Finally, we determine the appropriate data based on the spectra and
    # the preferred wavelength array.
    (
        interpolated_data,
        interpolated_uncertainty,
        interpolated_mask,
        interpolated_flags,
    ) = input_spectra.interpolate(
        wavelength=broadcast_wavelength,
        skip_mask=False,
        skip_flags=False,
    )
    # The data unit for the data and the like.
    broadcast_data_unit = input_spectra.data_unit

    # Now, we assemble the cube. We only need the spatial coverage of the cube
    # and broadcast our 1D spectra to the spatial dimensions. We do not
    # really care for the wavelength shape of the cube.
    x_dim, y_dim, __ = template_cube.data.shape
    wave_dim = broadcast_wavelength.shape[0]

    # If either the x or y dimension size is even, the center pixel is not
    # defined. Although we still attempt to put it in the center.
    if x_dim % 2 == 0 or y_dim % 2 == 0:
        # The image is even on one axis. We handle it according to the input
        # flag.
        if allow_even_center:
            # We attempt to find the center.
            logging.warning(
                warning_type=logging.AccuracyWarning,
                message=(
                    f"The image slice of the template cube is even: ({x_dim},"
                    f" {y_dim}). A best attempt at putting the spectra in the"
                    " center is attempted."
                ),
            )
        else:
            # We do not allow an even center, and so we raise.
            logging.critical(
                critical_type=logging.InputError,
                message=(
                    f"The image slice of the template cube is ({x_dim},"
                    f" {y_dim}). The image size must be odd for a defined"
                    " center."
                ),
            )
    # The center pixel is calculated similarly for both even and odd images.
    # Even images are just slightly off-centered.
    center_x = x_dim // 2
    center_y = y_dim // 2

    # We build the cube, using zero cube based on the provided shape to create
    # the broadcasts. We assign the appropriate location the data.
    # Data...
    broadcast_data = np.zeros((x_dim, y_dim, wave_dim))
    broadcast_data[center_x, center_y, :] = interpolated_data
    # Uncertainty...
    broadcast_uncertainty = np.zeros((x_dim, y_dim, wave_dim))
    broadcast_uncertainty[center_x, center_y, :] = interpolated_uncertainty
    # Mask...
    broadcast_mask = np.full((x_dim, y_dim, wave_dim), False)
    broadcast_mask[center_x, center_y, :] = interpolated_mask
    # Flags...
    broadcast_flags = np.full((x_dim, y_dim, wave_dim), 0, dtype=int)
    broadcast_flags[center_x, center_y, :] = interpolated_flags

    # Finally, we reconstruct the cube. We work on copies of the headers
    # just in case.
    spectra_header = input_spectra.header.copy()
    cube_header = template_cube.header.copy()
    cube_header.update(spectra_header)
    broadcast_header = cube_header

    # Building the new broadcasted cube. We use the template's cube's class
    # just in case it has been subclassed or something.
    template_cube_class = type(template_cube)
    broadcast_cube = template_cube_class(
        wavelength=broadcast_wavelength,
        data=broadcast_data,
        uncertainty=broadcast_uncertainty,
        wavelength_unit=broadcast_wavelength_unit,
        data_unit=broadcast_data_unit,
        mask=broadcast_mask,
        flags=broadcast_flags,
        header=broadcast_header,
    )

    # All done.
    return broadcast_cube
