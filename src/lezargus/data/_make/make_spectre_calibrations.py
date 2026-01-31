"""Make functions to create the SPECTRE calibration spectra.

This module contains all of the make functions to create calibration data
for SPECTRE. This data is either used by the simulator or the data reduction 
itself.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import astropy.table
import numpy as np

import lezargus
from lezargus.data._make import functionality
from lezargus.library import logging



def make_simulation_arclamp_spectrum(basename:str) -> hint.LezargusSpectrum:
    """Create an arc lamp spectrum for simulations.
    
    Parameters
    ----------
    basename : str
        The base filename of the data file which we read to obtain the 
        arclamp spectrum.
    
    Returns
    -------
    arc_lamp_spectrum : LezargusSpectrum
        The arc spectrum.
    """
    # Parsing the filename.
    arclamp_filename = functionality.find_data_filename(basename=basename)
    mrt_table = astropy.table.Table.read(arclamp_filename, format="ascii.mrt", )
    # Extracting the needed data.
    wavelength = mrt_table["wavelength"]
    flux = mrt_table["flux"]

    # The units of the arclamp spectrum as provided are not really 
    # conducive to a proper simulation, this needs to be fixed
    # properly and not just faked as is here.
    logging.error(error_type=logging.ToDoError, message=f"Arclamp flux units are in ct s^-1 (or DN s^-1) and should be converted to W / m. properly.")
    fake_arclamp_data_unit = lezargus.library.conversion.parse_astropy_unit(unit_input="W m^-1")         

    # Creating the spectrum object.
    arc_lamp_spectrum = lezargus.library.container.LezargusSpectrum(
        wavelength=wavelength,
        data=flux,
        uncertainty=None,
        wavelength_unit="m",
        data_unit=fake_arclamp_data_unit,
        spectral_scale=0,
        pixel_scale=None,
        slice_scale=None,
        mask=None,
        flags=None,
        header=None,
    )
    # All done.
    return arc_lamp_spectrum