"""Make functions to create the LezargusSpectrum objects for star spectra.

This module is just the part of the data making procedure to make the star
spectra objects. We only support a limited selection of stars because we also
load other metadata from the tables.
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


def make_standard_spectrum(basename: str) -> hint.LezargusSpectrum:
    """Load a spectrum data file and make a LezargusSpectrum class from it.

    Parameters
    ----------
    basename : str
        The basename of the internal data file of the spectrum. The paths are
        handled automatically.

    Returns
    -------
    spectrum : LezargusSpectrum
        The spectrum class.

    """
    # We need to load the data file.
    spectrum_table = load_spectrum_mrt_file(basename=basename)
    wavelength = np.asarray(spectrum_table["wavelength"])
    data = np.asarray(spectrum_table["data"])
    uncertainty = np.asarray(spectrum_table["uncertainty"])

    # We also need to load any potential metadata.
    header_table = load_spectrum_header_file()
    basename_tag = lezargus.library.path.get_filename_without_extension(
        pathname=basename,
    )
    try:
        header_keys = tuple(header_table["key"])
        header_values = tuple(header_table[basename_tag])
    except KeyError:
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                f"Internal header data for spectrum {basename_tag} does not"
                " exist in header table."
            ),
        )
        header = {}
    else:
        header = dict(zip(header_keys, header_values, strict=True))

    # Creating the spectrum object.
    spectrum = lezargus.library.container.LezargusSpectrum(
        wavelength=wavelength,
        data=data,
        uncertainty=uncertainty,
        wavelength_unit="m",
        data_unit="W m^-2 m^-1",
        spectral_scale=header.get("LZDSPECS", None),
        pixel_scale=header.get("LZDPIXPS", None),
        slice_scale=header.get("LZDSLIPS", None),
        mask=None,
        flags=None,
        header=header,
    )
    return spectrum


def load_spectrum_mrt_file(basename: str) -> hint.Table:
    """Load a AAS MRT spectrum file to a standard table format to use.

    Parameters
    ----------
    basename : str
        The MRT spectrum filename to load, paths are handled automatically.

    Returns
    -------
    spectrum_table : Table
        The Astropy table of the spectrum object.

    """
    # Parsing the filename.
    spectrum_filename = functionality.parse_basename(basename=basename)
    mrt_table = astropy.table.Table.read(spectrum_filename, format="ascii.mrt")
    # We format it to a standard table.
    wavelength_column = mrt_table["wavelength"]
    data_column = mrt_table["flux"]
    uncertainty_column = mrt_table["uncertainty"]
    # Parsing the table.
    spectrum_table = astropy.table.Table(
        [wavelength_column, data_column, uncertainty_column],
        names=("wavelength", "data", "uncertainty"),
    )
    return spectrum_table


def load_spectrum_header_file() -> hint.Table:
    """Load a header spectrum file to a standard table format to use.

    The spectrum header file is a single file with a set name.

    Parameters
    ----------
    None

    Returns
    -------
    header_table : Table
        The Astropy table of the header information for all star spectrums.

    """
    # We load in the table.
    header_filename = functionality.parse_basename(basename="star_header.dat")
    raw_header_table = astropy.table.Table.read(
        header_filename,
        format="ascii.fixed_width",
    )

    # We need to fix some of the data types to better format it to a proper
    # table.
    # We first need to generalize the data type.
    for columndex in tuple(raw_header_table.colnames):
        # We keep the string type of the key column.
        if columndex == "key":
            continue
        # Otherwise we convert.
        temp_new_column = raw_header_table[columndex].astype(object)
        raw_header_table[columndex] = temp_new_column

    for stardex, headerdex in raw_header_table.items():
        for index, valuedex in enumerate(headerdex):
            # Not all of the tests we have can be done via an "if" statement, so
            # we need to keep track if we already found a valid conversion.
            converted = False
            new_value = None

            # If the value is empty.
            if (not converted) and (valuedex is None or valuedex == "None"):
                converted = True
                new_value = None

            # If the value is a boolean type.
            if (not converted) and (valuedex in ("True", "False")):
                converted = True
                new_value = bool(valuedex)

            # If the value is a number.
            if not converted:
                try:
                    new_value = float(valuedex)
                except ValueError:
                    converted = False
                else:
                    converted = True

            # If none of the checks above find the proper type, we assume a
            # string.
            if (not converted) and isinstance(valuedex, str):
                new_value = str(valuedex)
                converted = True

            # Replacing the record with the new converted object.
            raw_header_table[stardex][index] = new_value

    # All done.
    header_table = raw_header_table
    return header_table
