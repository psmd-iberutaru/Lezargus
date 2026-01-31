"""Data objects needed for Lezargus are stored here.

We load and make the objects from accompanying data files.
"""

import lezargus
from lezargus.data import _make
from lezargus.library import logging

# Globals are how this entire module works, and it is readable as compared to
# using the globals dictionary. It is fine if a data statement has many
# statements as well, as long as they are related.
# ruff: noqa: PLW0603
# pylint: disable=global-variable-undefined, too-many-statements


def __initialize_data_all() -> None:
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
    __initialize_data_constants()
    __initialize_data_star_spectra()
    __initialize_data_photometric_filters()
    __initialize_data_detectors()
    __initialize_data_atmosphere_generators()
    __initialize_data_optic_efficiency_functions()
    __initialize_data_dispersion_patterns()
    __initialize_data_spectre_calibrations()


def __initialize_data_constants() -> None:
    """Initialize only the constant value data.

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

    # Mirror solid angles.
    global CONST_ENTRANCE_WINDOW_SOLID_ANGLE
    CONST_ENTRANCE_WINDOW_SOLID_ANGLE = _make.make_constant(
        key="CONST_ENTRANCE_WINDOW_SOLID_ANGLE",
    )

    global CONST_PRIMARY_MIRROR_SOLID_ANGLE
    CONST_PRIMARY_MIRROR_SOLID_ANGLE = _make.make_constant(
        key="CONST_PRIMARY_MIRROR_SOLID_ANGLE",
    )

    global CONST_SECONDARY_MIRROR_SOLID_ANGLE
    CONST_SECONDARY_MIRROR_SOLID_ANGLE = _make.make_constant(
        key="CONST_SECONDARY_MIRROR_SOLID_ANGLE",
    )

    # Pixel and slice sizes and widths.
    global CONST_SPECTRE_SLICES
    CONST_SPECTRE_SLICES = _make.make_constant(key="CONST_SPECTRE_SLICES")
    global CONST_VISIBLE_DETECTOR_SIZE
    CONST_VISIBLE_DETECTOR_SIZE = _make.make_constant(
        key="CONST_VISIBLE_DETECTOR_SIZE",
    )
    global CONST_NEARIR_DETECTOR_SIZE
    CONST_NEARIR_DETECTOR_SIZE = _make.make_constant(
        key="CONST_NEARIR_DETECTOR_SIZE",
    )
    global CONST_MIDIR_DETECTOR_SIZE
    CONST_MIDIR_DETECTOR_SIZE = _make.make_constant(
        key="CONST_MIDIR_DETECTOR_SIZE",
    )

    global CONST_VISIBLE_PIXEL_SIZE
    CONST_VISIBLE_PIXEL_SIZE = _make.make_constant(
        key="CONST_VISIBLE_PIXEL_SIZE",
    )
    global CONST_NEARIR_PIXEL_SIZE
    CONST_NEARIR_PIXEL_SIZE = _make.make_constant(key="CONST_NEARIR_PIXEL_SIZE")
    global CONST_MIDIR_PIXEL_SIZE
    CONST_MIDIR_PIXEL_SIZE = _make.make_constant(key="CONST_MIDIR_PIXEL_SIZE")

    # Cosmic ray parameters.
    global CONST_COSMIC_RAY_VALUE
    CONST_COSMIC_RAY_VALUE = _make.make_constant(key="CONST_COSMIC_RAY_VALUE")
    global CONST_COSMIC_RAY_RATE
    CONST_COSMIC_RAY_RATE = _make.make_constant(key="CONST_COSMIC_RAY_RATE")

    # Detector gain values.
    global CONST_VISIBLE_DETECTOR_GAIN
    CONST_VISIBLE_DETECTOR_GAIN = _make.make_constant(
        key="CONST_VISIBLE_DETECTOR_GAIN",
    )
    global CONST_NEARIR_DETECTOR_GAIN
    CONST_NEARIR_DETECTOR_GAIN = _make.make_constant(
        key="CONST_NEARIR_DETECTOR_GAIN",
    )
    global CONST_MIDIR_DETECTOR_GAIN
    CONST_MIDIR_DETECTOR_GAIN = _make.make_constant(
        key="CONST_MIDIR_DETECTOR_GAIN",
    )

    # All done.
    return


def __initialize_data_star_spectra() -> None:
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


def __initialize_data_photometric_filters() -> None:  # noqa: PLR0915
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
                critical_type=logging.DevelopmentError,
                message="Standard A0V star is not a LezargusSpectrum.",
            )
    except NameError:
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

    # TYCHO-2 Filters
    global FILTER_TYCHO2_BT
    FILTER_TYCHO2_BT = _make.make_vega_photometric_filter(
        basename="filter_TYCHO2_BT.dat",
    )
    FILTER_TYCHO2_BT.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_TYCHO2_VT
    FILTER_TYCHO2_VT = _make.make_vega_photometric_filter(
        basename="filter_TYCHO2_VT.dat",
    )
    FILTER_TYCHO2_VT.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    # GAIA Filters
    global FILTER_GAIA_BP
    FILTER_GAIA_BP = _make.make_vega_photometric_filter(
        basename="filter_GAIA_BP.dat",
    )
    FILTER_GAIA_BP.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_GAIA_G
    FILTER_GAIA_G = _make.make_vega_photometric_filter(
        basename="filter_GAIA_G.dat",
    )
    FILTER_GAIA_G.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_GAIA_RP
    FILTER_GAIA_RP = _make.make_vega_photometric_filter(
        basename="filter_GAIA_RP.dat",
    )
    FILTER_GAIA_RP.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_GAIA_RVS
    FILTER_GAIA_RVS = _make.make_vega_photometric_filter(
        basename="filter_GAIA_RVS.dat",
    )
    FILTER_GAIA_RVS.add_standard_star_spectrum(
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

    # WISE 1 2 3 4 Filters.
    global FILTER_WISE_1
    FILTER_WISE_1 = _make.make_vega_photometric_filter(
        basename="filter_WISE_1.dat",
    )
    FILTER_WISE_1.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_WISE_2
    FILTER_WISE_2 = _make.make_vega_photometric_filter(
        basename="filter_WISE_2.dat",
    )
    FILTER_WISE_2.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_WISE_3
    FILTER_WISE_3 = _make.make_vega_photometric_filter(
        basename="filter_WISE_3.dat",
    )
    FILTER_WISE_3.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )

    global FILTER_WISE_4
    FILTER_WISE_4 = _make.make_vega_photometric_filter(
        basename="filter_WISE_4.dat",
    )
    FILTER_WISE_4.add_standard_star_spectrum(
        spectrum=STAR_A0V,
        magnitude=0,
        magnitude_uncertainty=0,
    )


def __initialize_data_detectors() -> None:
    """Initialize only the detector specifications.

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
    global SPECTRE_DETECTOR_VISIBLE
    SPECTRE_DETECTOR_VISIBLE = _make.make_detector()

    global SPECTRE_DETECTOR_NEARIR
    SPECTRE_DETECTOR_NEARIR = _make.make_detector()

    global SPECTRE_DETECTOR_MIDIR
    SPECTRE_DETECTOR_MIDIR = _make.make_detector()

    # All done.
    return


def __initialize_data_atmosphere_generators() -> None:
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
    ATM_RADIANCE_GEN = _make.make_atmosphere_radiance_generator(
        basename="psg_telluric_radiance.dat",
    )


def __initialize_data_optic_efficiency_functions() -> None:
    """Initialize only the optic efficiency function spectrums.

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
    # Creating the IRTF efficiency functions for the primary and secondary
    # mirror.
    global IRTF_EFFICIENCY_PRIMARY
    IRTF_EFFICIENCY_PRIMARY = _make.make_optic_efficiency(
        basename="efficiency_irtf_primary_mirror.dat",
    )

    global IRTF_EFFICIENCY_SECONDARY
    IRTF_EFFICIENCY_SECONDARY = _make.make_optic_efficiency(
        basename="efficiency_irtf_secondary_mirror.dat",
    )

    # Creating the SPECTRE entrance window efficiency.
    global SPECTRE_EFFICIENCY_WINDOW
    SPECTRE_EFFICIENCY_WINDOW = _make.make_optic_efficiency(
        basename="spectre_efficiency_entrance_window.dat",
    )

    # Creating the SPECTRE collimator mirror efficiency.
    global SPECTRE_EFFICIENCY_COLLIMATOR
    SPECTRE_EFFICIENCY_COLLIMATOR = _make.make_optic_efficiency(
        basename="spectre_efficiency_collimator_mirror.dat",
    )

    # Creating the SPECTRE camera mirror efficiency.
    global SPECTRE_EFFICIENCY_CAMERA
    SPECTRE_EFFICIENCY_CAMERA = _make.make_optic_efficiency(
        basename="spectre_efficiency_camera_mirror.dat",
    )

    # Creating the SPECTRE IFU image slicer efficiency.
    global SPECTRE_EFFICIENCY_IMAGE_SLICER
    SPECTRE_EFFICIENCY_IMAGE_SLICER = _make.make_optic_efficiency(
        basename="spectre_efficiency_image_slicer.dat",
    )

    # Creating the SPECTRE IFU pupil mirror efficiency.
    global SPECTRE_EFFICIENCY_PUPIL_MIRROR
    SPECTRE_EFFICIENCY_PUPIL_MIRROR = _make.make_optic_efficiency(
        basename="spectre_efficiency_pupil_mirror.dat",
    )

    # Creating the SPECTRE dichroic efficiencies.
    global SPECTRE_EFFICIENCY_DICHROIC_VISIBLE
    SPECTRE_EFFICIENCY_DICHROIC_VISIBLE = _make.make_optic_efficiency(
        basename="spectre_efficiency_dichroic_visible.dat",
    )
    global SPECTRE_EFFICIENCY_DICHROIC_NEARIR
    SPECTRE_EFFICIENCY_DICHROIC_NEARIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_dichroic_nearir.dat",
    )
    global SPECTRE_EFFICIENCY_DICHROIC_MIDIR
    SPECTRE_EFFICIENCY_DICHROIC_MIDIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_dichroic_midir.dat",
    )

    # Creating the SPECTRE relay mirror efficiencies.
    global SPECTRE_EFFICIENCY_RELAY_VISIBLE
    SPECTRE_EFFICIENCY_RELAY_VISIBLE = _make.make_optic_efficiency(
        basename="spectre_efficiency_relay_mirror_visible.dat",
    )
    global SPECTRE_EFFICIENCY_RELAY_NEARIR
    SPECTRE_EFFICIENCY_RELAY_NEARIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_relay_mirror_nearir.dat",
    )
    global SPECTRE_EFFICIENCY_RELAY_MIDIR
    SPECTRE_EFFICIENCY_RELAY_MIDIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_relay_mirror_midir.dat",
    )

    # Creating the SPECTRE prism efficiencies.
    global SPECTRE_EFFICIENCY_PRISM_BK7
    SPECTRE_EFFICIENCY_PRISM_BK7 = _make.make_optic_efficiency(
        basename="spectre_efficiency_prism_bk7.dat",
    )
    global SPECTRE_EFFICIENCY_PRISM_SILICA
    SPECTRE_EFFICIENCY_PRISM_SILICA = _make.make_optic_efficiency(
        basename="spectre_efficiency_prism_silica.dat",
    )
    global SPECTRE_EFFICIENCY_PRISM_ZNSE
    SPECTRE_EFFICIENCY_PRISM_ZNSE = _make.make_optic_efficiency(
        basename="spectre_efficiency_prism_znse.dat",
    )
    global SPECTRE_EFFICIENCY_PRISM_SAPPHIRE
    SPECTRE_EFFICIENCY_PRISM_SAPPHIRE = _make.make_optic_efficiency(
        basename="spectre_efficiency_prism_sapphire.dat",
    )

    # Creating the SPECTRE fold mirror efficiencies.
    global SPECTRE_EFFICIENCY_FOLD_VISIBLE
    SPECTRE_EFFICIENCY_FOLD_VISIBLE = _make.make_optic_efficiency(
        basename="spectre_efficiency_fold_mirror_visible.dat",
    )
    global SPECTRE_EFFICIENCY_FOLD_NEARIR
    SPECTRE_EFFICIENCY_FOLD_NEARIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_fold_mirror_nearir.dat",
    )
    global SPECTRE_EFFICIENCY_FOLD_MIDIR
    SPECTRE_EFFICIENCY_FOLD_MIDIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_fold_mirror_midir.dat",
    )

    # Creating the SPECTRE detector quantum efficiencies.
    global SPECTRE_EFFICIENCY_CCD_VISIBLE
    SPECTRE_EFFICIENCY_CCD_VISIBLE = _make.make_optic_efficiency(
        basename="spectre_efficiency_ccd_qe.dat",
    )
    global SPECTRE_EFFICIENCY_H2RG_NEARIR
    SPECTRE_EFFICIENCY_H2RG_NEARIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_h2rg25_qe.dat",
    )
    global SPECTRE_EFFICIENCY_H2RG_MIDIR
    SPECTRE_EFFICIENCY_H2RG_MIDIR = _make.make_optic_efficiency(
        basename="spectre_efficiency_h2rg40_qe.dat",
    )

    # All done
    return


def __initialize_data_dispersion_patterns() -> None:
    """Initialize only the spectral/detector dispersion patterns.

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

    # The SPECTRE dispersion pattern.
    global SPECTRE_DISPERSION
    SPECTRE_DISPERSION = _make.make_spectre_dispersion_pattern(
        basename="spectre_slice_dispersion.dat",
    )


def __initialize_data_spectre_calibrations() -> None:
    """Initialize only the SPECTRE calibrations.

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

    # Arc lamp simulation spectra.
    # Visible class...
    global SPECTRE_ARCLAMP_SIMULATION_VISIBLE
    SPECTRE_ARCLAMP_SIMULATION_VISIBLE = _make.make_simulation_arclamp_spectrum(
        basename="visible_hgar_arclamp.dat",
    )
    # NearIR class...
    global SPECTRE_ARCLAMP_SIMULATION_NEARIR
    SPECTRE_ARCLAMP_SIMULATION_NEARIR = _make.make_simulation_arclamp_spectrum(
        basename="nearir_spex_sxd_arclamp.dat",
    )
    # MidIR class...
    global SPECTRE_ARCLAMP_SIMULATION_MIDIR
    SPECTRE_ARCLAMP_SIMULATION_MIDIR = _make.make_simulation_arclamp_spectrum(
        basename="midir_spex_lxds_arclamp.dat",
    )




__initialize_data_all()
