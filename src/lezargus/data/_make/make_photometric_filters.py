"""Make functions to create the PhotometricFilter objects for filters.

This module is just the part of the data making procedure to make the
photometric filter objects. We only support a limited selection of filters as
implemented by the data files. Open a new issue to add more filters if desired.
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


def make_ab_photometric_filter(basename: str) -> hint.PhotometricABFilter:
    """Load a photometric filter file to make a PhotometricABFilter class.

    This also technically handles ST filters, but all of the ST filters should
    be reformulated in an AB form.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the AB-based photometric
        filter. The paths are handled automatically.

    Returns
    -------
    ab_filter : PhotometricVegaFilter
        The AB-based photometric filter class.

    """
    lezargus.library.wrapper.do_nothing(basename)
    logging.critical(
        critical_type=logging.ToDoError,
        message=(
            "AB photometric filter loaders not completed, no AB filter"
            " implemented so far."
        ),
    )
    return lezargus.library.container.PhotometricABFilter()


def make_vega_photometric_filter(basename: str) -> hint.PhotometricVegaFilter:
    """Load a photometric filter file to make a PhotometricVegaFilter class.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the Vega-based photometric
        filter. The paths are handled automatically.

    Returns
    -------
    vega_filter : PhotometricVegaFilter
        The Vega-based photometric filter class.

    """
    # We need to load the data file, and the data in general.
    photometric_filter_table = load_photometric_filter_mrt_file(
        basename=basename,
    )
    wavelength = np.asarray(photometric_filter_table["wavelength"])
    transmission = np.asarray(photometric_filter_table["transmission"])

    # We load the filter, leveraging the Lezargus container to do the
    # conversion for us. By default, all of the Vega filters are in energy
    # integrating mode.
    _photometric_class = lezargus.library.container.PhotometricVegaFilter
    vega_filter = _photometric_class.from_energy_transmission(
        wavelength=wavelength,
        energy_transmission=transmission,
        wavelength_unit="m",
    )
    return vega_filter


def load_photometric_filter_mrt_file(basename: str) -> hint.Table:
    """Load a AAS MRT photometric filter file to a standard table format.

    Parameters
    ----------
    basename : str
        The MRT photometric filter basename to load, paths are handled
        automatically.

    Returns
    -------
    photometric_filter_table : Table
        The Astropy table of the photometric filter object.

    """
    # We parse the data file and load it.
    filter_filename = functionality.parse_basename(basename=basename)
    mrt_table = astropy.table.Table.read(filter_filename, format="ascii.mrt")
    # We format it to a standard table.
    wavelength_column = mrt_table["wavelength"]
    data_column = mrt_table["transmission"]
    # Parsing the table.
    spectrum_table = astropy.table.Table(
        [wavelength_column, data_column],
        names=("wavelength", "transmission"),
    )
    return spectrum_table
