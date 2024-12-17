"""Make functions to create the efficiency functions for different optics.

This module is created to making efficiency functions (or efficiency spectra)
for a wide array of things. We package these as typical LezargusSpectrum
objects as they are a convenient way to store them.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.table

import lezargus
from lezargus.data._make import functionality


def make_optic_efficiency(basename: str) -> hint.LezargusSpectrum:
    """Load the a single optic efficiency spectrum file.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the optic efficiency file.
        The paths are handled automatically.

    Returns
    -------
    efficiency_spectrum : LezargusSpectrum
        The optic efficiency function spectrum.

    """
    # Parsing the filename.
    optic_filename = functionality.parse_basename(basename=basename)
    mrt_table = astropy.table.Table.read(optic_filename, format="ascii.mrt")
    # Extracting the needed data.
    wavelength = mrt_table["wavelength"]
    efficiency = mrt_table["efficiency"]

    # Creating the spectrum object.
    efficiency_spectrum = lezargus.library.container.LezargusSpectrum(
        wavelength=wavelength,
        data=efficiency,
        uncertainty=None,
        wavelength_unit="m",
        data_unit="",
        spectral_scale=0,
        pixel_scale=None,
        slice_scale=None,
        mask=None,
        flags=None,
        header=None,
    )
    # All done.
    return efficiency_spectrum
