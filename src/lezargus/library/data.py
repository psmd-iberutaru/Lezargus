"""Data file functions.

This file deals with the loading in and saving of data files which are in
the /data/ directory of Lezargus. Moreover, the contents of the data
are accessed using attributes of this module.
"""

import os

import lezargus
from lezargus import library

# We need to get the actual directory of the data.
MODULE_DATA_DIRECTORY = os.path.join(
    library.config.MODULE_INSTALLATION_PATH,
    "data",
)


def initialize_data_files() -> None:
    """Create all of the data files and instances of classes.

    This function creates all of the data objects which represent all of the
    data and saves it to this module. This must be done in a function, and
    called by the initialization of the module, to avoid import errors and
    dependency issues.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    data_files = {}
    # Loading the stars, often used as standard stars.
    data_files["STAR_16CYGB"] = lezargus.container.LezargusSpectra.read_fits_file(
        filename=library.path.merge_pathname(
            directory=MODULE_DATA_DIRECTORY,
            filename="star_spectra_16CygB",
            extension="fits",
        ),
    )
    data_files["STAR_109VIR"] = lezargus.container.LezargusSpectra.read_fits_file(
        filename=library.path.merge_pathname(
            directory=MODULE_DATA_DIRECTORY,
            filename="star_spectra_109Vir",
            extension="fits",
        ),
    )
    data_files["STAR_SUN"] = lezargus.container.LezargusSpectra.read_fits_file(
        filename=library.path.merge_pathname(
            directory=MODULE_DATA_DIRECTORY,
            filename="star_spectra_Sun",
            extension="fits",
        ),
    )
    data_files["STAR_VEGA"] = lezargus.container.LezargusSpectra.read_fits_file(
        filename=library.path.merge_pathname(
            directory=MODULE_DATA_DIRECTORY,
            filename="star_spectra_Vega",
            extension="fits",
        ),
    )

    # Loading the filters.

    # Finally, applying the data to this module.
    globals().update(data_files)