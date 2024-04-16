"""Data file functions.

This file deals with the loading in and saving of data files which are in
the /data/ directory of Lezargus. Moreover, the contents of the data
are accessed using attributes of this module.

Also, custom functions are provided to make things which are similarly
contained in the data directory. You can find all of these functions under the
`custom_*` namespace.
"""

import numpy as np

import lezargus
from lezargus.library import hint
from lezargus.library import logging


def add_data_object(name: str, data: object, force: bool = False) -> None:
    """Add a data variable to the library data module.

    If the addition of a new data object with the same name as one already
    present, we raise an error as these values are technically constants.


    Parameters
    ----------
    name : str
        The name of the data object to be added, it will be coerced to be all
        capitalized as it ought to be effectively a constant.
    data : object
        The data object to be added to the data module.
    force : bool
        Force the loading of the data object regardless of any critical errors.

    """
    # We fix the name.
    name = name.upper()
    # Check the current global namespace for any already-existing data objects.
    for namedex in globals():
        if name == namedex:
            # A name conflict is detected, so, we need to give an error. If
            # we are not forcing it to be loaded, we use a critical error
            # instead.
            if force:
                logging.error(
                    error_type=logging.InputError,
                    message=(
                        f"Object name {name} already exists in the data module."
                        " We force its overwriting."
                    ),
                )
            else:
                logging.critical(
                    critical_type=logging.InputError,
                    message=(
                        f"Object name {name} already exists in the data module."
                        " We cannot assign a name to the data object."
                    ),
                )
    # We add the name and the object.
    logging.debug(
        message=(
            f"Added {name} to the lezargus.library.data module; it is of type"
            f" {type(data)}."
        ),
    )
    globals().update({name: data})


def _zero_buffer_custom_filters(
    wavelength: hint.ndarray,
    transmission: hint.ndarray,
) -> tuple[hint.ndarray, hint.ndarray]:
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
    delta = 10 ** np.floor(np.log10(spacing)) * 1e-5
    # The actual points.
    extra_zero_blue_wave = (
        np.min(wavelength) - ((np.arange(n_buffer_points)) * spacing) - delta
    )
    extra_zero_red_wave = (
        np.max(wavelength) + ((np.arange(n_buffer_points)) * spacing) + delta
    )
    extra_zero_wave = np.append(extra_zero_blue_wave, extra_zero_red_wave)
    extra_zero_trans = np.zeros_like(extra_zero_wave)
    # Now adding them to the original spectra.
    new_zero_wavelength = np.append(wavelength, extra_zero_wave)
    new_zero_transmission = np.append(transmission, extra_zero_trans)
    # Sorting as well.
    sort_index = np.argsort(new_zero_wavelength)
    zero_wavelength = new_zero_wavelength[sort_index]
    zero_transmission = new_zero_transmission[sort_index]
    # All done.
    return zero_wavelength, zero_transmission


def custom_rectangular_filter(
    lower_limit: float,
    upper_limit: float,
) -> hint.LezargusSpectrum:
    """Make a custom rectangular filter profile.

    Parameters
    ----------
    lower_limit : float
        The lower limit of the rectangular filter. This value is typically
        a wavelength, in meters.
    upper_limit : float
        The upper limit of the rectangular filter. This value is typically
        a wavelength, in meters.

    Returns
    -------
    rectangular_filter : LezargusSpectrum
        The filter, as defined.

    """
    # Wavelength and data. The 100 data points are arbitrary but we think it
    # is enough.
    n_data_points = 100
    filter_wave = np.linspace(lower_limit, upper_limit, n_data_points)
    filter_trans = np.ones_like(filter_wave)

    # Customarily, filters have a little bit of data outside their band pass
    # for zeros.
    buffer_filter_wave, buffer_filter_trans = _zero_buffer_custom_filters(
        wavelength=filter_wave,
        transmission=filter_trans,
    )
    # Now we construct the filter object.
    rectangular_filter = lezargus.container.LezargusSpectrum(
        wavelength=buffer_filter_wave,
        data=buffer_filter_trans,
        uncertainty=None,
        wavelength_unit="m",
        data_unit="",
        header={"LZO_NAME": "Custom_rect_photon"},
    )
    # All done.
    return rectangular_filter
