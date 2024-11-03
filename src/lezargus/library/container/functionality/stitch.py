"""Stitch spectra, images, and cubes together.

Stitching spectra, images, and cubes consistently, while keeping all of the
pitfalls in check, is not trivial. We group these three stitching functions,
and the required spin-off functions, here.
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


def calculate_spectra_scale_factor(
    base_spectrum: hint.LezargusSpectrum,
    input_spectrum: hint.LezargusSpectrum,
    bounds: tuple[float, float] = (-np.inf, +np.inf),
) -> tuple[float, float]:
    """Find the scale factor to scale one overlapping spectrum to another.

    This implementation relies on
    py:func:`lezargus.library.stitch.calculate_spectra_scale_factor`

    Parameter
    ---------
    base_spectrum : LezargusSpectrum
        The spectrum class for the base spectrum. The units of both spectra
        should be consistent.
    input_spectrum : LezargusSpectrum
        The spectrum class for the input spectrum which this scale factor is
        being calculated for. The units of both spectra should be consistent.
    bounds : tuple, default = (-np.inf, +np.inf)
        An additional set of wavelength bounds to specify the limits of the
        overlap which we use to determine the scale factor. Format is
        (minimum, maximum). Must be in the same units as the base and input
        wavelengths.

    Returns
    -------
    scale_factor : float
        The scale factor to scale the input data to match the base data.
    scale_uncertainty : float
        The uncertainty in the scale factor. This is usually not relevant.

    """
    # We need to make sure the units are consistent.
    if base_spectrum.wavelength_unit != input_spectrum.wavelength_unit:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                f"Base wavelength unit {base_spectrum.wavelength_unit} not the"
                f" same as {input_spectrum.wavelength_unit}."
            ),
        )
    if base_spectrum.data_unit != input_spectrum.data_unit:
        logging.warning(
            warning_type=logging.AccuracyWarning,
            message=(
                f"Base data unit {base_spectrum.data_unit} not the same as"
                f" {input_spectrum.data_unit}."
            ),
        )

    # The internal implementation used just normal arrays, we keep that and
    # just extract the parts we need from the main classes. The old
    # implementation is completely valid.
    base_wavelength = base_spectrum.wavelength
    base_data = base_spectrum.data
    base_uncertainty = base_spectrum.uncertainty
    input_wavelength = input_spectrum.wavelength
    input_data = input_spectrum.data
    input_uncertainty = input_spectrum.uncertainty

    scale_factor, scale_uncertainty = (
        lezargus.library.stitch.calculate_spectra_scale_factor(
            base_wavelength=base_wavelength,
            base_data=base_data,
            input_wavelength=input_wavelength,
            input_data=input_data,
            base_uncertainty=base_uncertainty,
            input_uncertainty=input_uncertainty,
            bounds=bounds,
        )
    )

    return scale_factor, scale_uncertainty
