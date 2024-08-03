"""Data objects needed for Lezargus are stored here.

We load and make the objects from accompanying data files.
"""

import lezargus
from lezargus.data import _make
from lezargus.library import logging

# Globals are how this entire module works, and it is readable as compared to
# using the globals dictionary.
# ruff: noqa: PLW0603
# pylint: disable=global-variable-undefined


def __init_data_all() -> None:
    """Initialize the all of the data objects.

    We wrap the initialization of the data objects in a function so we can
    handle it with a more fine grained approach. The use of the global keyword
    enables the objects to be the global space of this module anyways.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # If the initialization of the data is to be skipped.
    if lezargus.config.INTERNAL_DEBUG_SKIP_LOADING_DATA_FILES:
        return

    # Loading the data in the proper order.
    __init_data_spectra()
    __init_data_photometric_filters()
    __init_data_atmosphere_generators()


def __init_data_spectra() -> None:
    """Initialize only the standard spectral data.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # If the initialization of the data is to be skipped.
    if lezargus.config.INTERNAL_DEBUG_SKIP_LOADING_DATA_FILES:
        return

    # Otherwise...

    # Creating all of the standard (star) spectra objects.
    global STAR_16CYGB
    STAR_16CYGB = _make.make_standard_spectrum(
        basename="star_spectra_16CygB.dat",
    )

    global STAR_109VIR
    STAR_109VIR = _make.make_standard_spectrum(
        basename="star_spectra_109Vir.dat",
    )

    global STAR_A0V
    STAR_A0V = _make.make_standard_spectrum(basename="star_spectra_A0V.dat")

    global STAR_SUN
    STAR_SUN = _make.make_standard_spectrum(basename="star_spectra_Sun.dat")

    global STAR_VEGA
    STAR_VEGA = _make.make_standard_spectrum(basename="star_spectra_Vega.dat")

    # All done.
    return


def __init_data_photometric_filters() -> None:
    """Initialize only the photometric filters.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # If the initialization of the data is to be skipped.
    if lezargus.config.INTERNAL_DEBUG_SKIP_LOADING_DATA_FILES:
        return

    # Otherwise...

    # If the standard Vega spectrum does not exist, we cannot actually
    # load the filters. The spectrum needs to be done first.
    try:
        if not isinstance(
            STAR_A0V,
            lezargus.library.container.LezargusSpectrum,
        ):
            logging.critical(
                critical_type=logging.ExpectedCaughtError,
                message="Standard A0V star is not a LezargusSpectrum.",
            )
    except (NameError, logging.ExpectedCaughtError):
        logging.critical(
            critical_type=logging.DevelopmentError,
            message=(
                "Standard A0V spectrum data does not exist in data module."
                " Cannot properly load photometric filters."
            ),
        )

    # Creating all of the photometric filter objects.
    # Johnson U B V filters.
    global FILTER_JOHNSON_U
    FILTER_JOHNSON_U = _make.make_vega_photometric_filter(
        basename="filter_Johnson_U.dat",
    )
    FILTER_JOHNSON_U.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_JOHNSON_B
    FILTER_JOHNSON_B = _make.make_vega_photometric_filter(
        basename="filter_Johnson_B.dat",
    )
    FILTER_JOHNSON_B.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_JOHNSON_V
    FILTER_JOHNSON_V = _make.make_vega_photometric_filter(
        basename="filter_Johnson_V.dat",
    )
    FILTER_JOHNSON_V.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    # 2MASS J H Ks filters.
    global FILTER_2MASS_J
    FILTER_2MASS_J = _make.make_vega_photometric_filter(
        basename="filter_2MASS_J.dat",
    )
    FILTER_2MASS_J.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_2MASS_H
    FILTER_2MASS_H = _make.make_vega_photometric_filter(
        basename="filter_2MASS_H.dat",
    )
    FILTER_2MASS_H.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_2MASS_KS
    FILTER_2MASS_KS = _make.make_vega_photometric_filter(
        basename="filter_2MASS_Ks.dat",
    )
    FILTER_2MASS_KS.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )


def __init_data_atmosphere_generators() -> None:
    """Initialize only the atmospheric generators.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # If the initialization of the data is to be skipped.
    if lezargus.config.INTERNAL_DEBUG_SKIP_LOADING_DATA_FILES:
        return

    # Otherwise...

    # Creating the atmospheric transmission and radiance generators.
    global ATM_TRANS_GEN
    ATM_TRANS_GEN = _make.make_atmosphere_transmission_generator(
        basename="psg_telluric_transmission.dat",
    )

    global ATM_RADIANCE_GEN
    ATM_RADIANCE_GEN = _make.make_atmosphere_transmission_generator(
        basename="psg_telluric_radiance.dat",
    )


__init_data_all()
