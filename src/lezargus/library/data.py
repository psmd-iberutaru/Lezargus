"""Data file functions.

This file deals with the loading in and saving of data files which are in
the /data/ directory of Lezargus. Moreover, the contents of the data
are accessed using attributes of this module.

Also, custom functions are provided to make things which are similarly 
contained in the data directory. You can find all of these functions under the
`custom_*` namespace.
"""

import os

import numpy as np

from lezargus import library
from lezargus.library import logging
from lezargus.library import hint
import lezargus

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
    logging.debug(message="Loading standard star data for data initialization.")
    data_files["STAR_16CYGB"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="star_spectra_16CygB",
                extension="fits",
            ),
        )
    )
    data_files["STAR_109VIR"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="star_spectra_109Vir",
                extension="fits",
            ),
        )
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
    data_files["STAR_A0V"] = lezargus.container.LezargusSpectra.read_fits_file(
        filename=library.path.merge_pathname(
            directory=MODULE_DATA_DIRECTORY,
            filename="star_spectra_A0V",
            extension="fits",
        ),
    )

    # Loading the filters.
    # Johnson...
    logging.debug(
        message="Loading Johnson photometric filters for data initialization.",
    )
    data_files["FILTER_JOHNSON_U_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_U_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_JOHNSON_B_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_B_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_JOHNSON_V_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_V_photon",
                extension="fits",
            ),
        )
    )
    # Gaia...
    logging.debug(
        message="Loading Gaia photometric filters for data initialization.",
    )
    data_files["FILTER_GAIA_GG_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GG_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_GAIA_GB_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GB_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_GAIA_GR_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GR_photon",
                extension="fits",
            ),
        )
    )
    # 2MASS...
    logging.debug(
        message="Loading 2MASS photometric filters for data initialization.",
    )
    data_files["FILTER_2MASS_J_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_J_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_2MASS_H_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_H_photon",
                extension="fits",
            ),
        )
    )
    data_files["FILTER_2MASS_KS_PHOTON"] = (
        lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_Ks_photon",
                extension="fits",
            ),
        )
    )

    # We next calculate filter zero points for each filter set.
    logging.debug(
        message=(
            "Calculating photometric filter zero points and errors for data"
            " initialization."
        ),
    )
    # Johnson...
    (
        data_files["ZERO_POINT_VEGA_JOHNSON_U"],
        data_files["ZERO_POINT_VEGA_JOHNSON_U_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_JOHNSON_U_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_J_U"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_J_U"],
    )
    (
        data_files["ZERO_POINT_VEGA_JOHNSON_B"],
        data_files["ZERO_POINT_VEGA_JOHNSON_B_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_JOHNSON_B_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_J_B"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_J_B"],
    )
    (
        data_files["ZERO_POINT_VEGA_JOHNSON_V"],
        data_files["ZERO_POINT_VEGA_JOHNSON_V_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_JOHNSON_V_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_J_V"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_J_V"],
    )
    # Gaia...

    # 2MASS
    (
        data_files["ZERO_POINT_VEGA_2MASS_J"],
        data_files["ZERO_POINT_VEGA_2MASS_J_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_2MASS_J_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_2_J"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_2_J"],
    )
    (
        data_files["ZERO_POINT_VEGA_2MASS_H"],
        data_files["ZERO_POINT_VEGA_2MASS_H_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_2MASS_H_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_2_H"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_2_H"],
    )
    (
        data_files["ZERO_POINT_VEGA_2MASS_KS"],
        data_files["ZERO_POINT_VEGA_2MASS_KS_ERROR"],
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=data_files["FILTER_2MASS_KS_PHOTON"],
        standard_spectra=data_files["STAR_A0V"],
        standard_filter_magnitude=data_files["STAR_A0V"].header["LZOM_2KS"],
        standard_filter_uncertainty=data_files["STAR_A0V"].header["LZOU_2KS"],
    )

    # Finally, applying the data to this module.
    globals().update(data_files)
    logging.info(
        message=(
            "Initialized Lezargus data have been loaded into to the data"
            " module."
        ),
    )


def _zero_buffer_custom_filters(wavelength:hint.ndarray, transmission:hint.ndarray) -> tuple[hint.ndarray, hint.ndarray]:
    """Create a zero transmission buffer on either side of the filter.
    
    This function is a convenience function for creating tail ends of created
    custom filter profiles with zero transmission, as expected.

    Parameters
    ----------
    wavelength : ndarray
        The original wavelength of the filter.
    transmission : ndarray
        The original transmission wavelength of the filter.

    Returns
    -------
    zero_wavelength : ndarray
        The wavelength, with added points for the zero section.
    zero_transmission : ndarray
        The transmission, with added points for the zero section, of zero.
    """
    # The number of buffer points we have to make on each side.
    n_buffer_points = 5 + 1
    # Creating the buffer points.
    spacing = np.ptp(wavelength) / n_buffer_points
    # This helps the profiles be sharp by have a new data point very close to
    # the defined ones.
    delta = 10**np.floor(np.log10(spacing)) * 1e-5
    # The actual points.
    extra_zero_blue_wave = np.min(wavelength) - ((np.arange(n_buffer_points)) * spacing) - delta
    extra_zero_red_wave = np.max(wavelength) + ((np.arange(n_buffer_points)) * spacing) + delta
    extra_zero_wave = np.append(extra_zero_blue_wave, extra_zero_red_wave)
    extra_zero_trans = np.zeros_like(extra_zero_wave)
    # Now adding them to the original spectra.
    new_zero_wavelength = np.append(wavelength, extra_zero_wave)
    new_zero_transmission = np.append(transmission, extra_zero_trans)
    # Sorting as well.
    sort_index = np.argsort(new_zero_wavelength)
    zero_wavelength = new_zero_wavelength[sort_index]
    zero_transmission =new_zero_transmission[sort_index]
    # All done.
    return zero_wavelength, zero_transmission

def custom_rectangular_filter(lower_limit:float, upper_limit:float) -> hint.LezargusSpectra:
    """Make a custom rectangular filter profile.
    
    Parameters
    ----------
    lower_limit : float
        The lower limit of the rectangular filter. This value is typically
        a wavelength.
    upper_limit : float
        The upper limit of the rectangular filter. This value is typically
        a wavelength.

    Returns
    -------
    rectangular_filter : LezargusSpectra
        The filter, as defined.
    """
    # Wavelength and data. The 100 data points are arbitrary but we think it 
    # is enough.
    n_data_points = 100
    filter_wave = np.linspace(lower_limit, upper_limit, n_data_points)
    filter_trans = np.ones_like(filter_wave)

    # Customarily, filters have a little bit of data outside their band pass 
    # for zeros.
    buffer_filter_wave, buffer_filter_trans = _zero_buffer_custom_filters(wavelength=filter_wave, transmission=filter_trans)
    # Now we construct the filter object.
    rectangular_filter = lezargus.container.LezargusSpectra(
    wavelength=buffer_filter_wave,
    data=buffer_filter_trans,
    uncertainty=None,
    wavelength_unit="um",
    data_unit="",
    header={
        "LZO_NAME":"Custom_rect_photon"
    }
)
    # All done.
    return rectangular_filter
