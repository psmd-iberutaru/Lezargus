"""Module, file, and data initialization routines of Lezargus.

Everything and anything which initializes Lezargus, that is separate from
Python loading this module, is done here.
"""

import glob
import os
import sys
import uuid

import astropy.table
import numpy as np

import lezargus
from lezargus.library import logging


def initialize(*args: tuple, **kwargs: object) -> None:
    """Initialize the Lezargus module and all its parts.

    This initialization function should be the very first thing that is done
    when the module is loaded. However, we create this function (as opposed to
    doing it on load) to be explicit on the load times for the module, to
    avoid circular dependencies, and to prevent logging when only importing
    the module.

    The order of the initialization is important and we take care of it here.
    If you want to want to initialize smaller sections independently, you
    may use the functions within the :py:mod:`lezargus.initialize` module.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        Keyword arguments to be passed to all other initialization functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Load in the default configuration file.
    initialize_configuration(**kwargs)

    # Load the logging outputs.
    initialize_logging_outputs(**kwargs)

    # All of the initializations below have logging.

    # Loading and creating the needed temporary directories.
    initialize_temporary_directory(**kwargs)

    # Load all of the data files for Lezargus.
    initialize_data_all(**kwargs)


def initialize_configuration(*args: tuple, **kwargs: object) -> None:
    """Initialize the default configuration file.

    This function forces the reading and applying of the default
    configuration file. Note, this should not called when a user configuration
    file has already been provided.


    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Load the default configuration parameters. The user's configurations
    # should overwrite these when supplied.
    lezargus.library.config.load_configuration_file(
        filename=lezargus.library.path.merge_pathname(
            directory=lezargus.library.config.INTERNAL_MODULE_INSTALLATION_PATH,
            filename="configuration",
            extension="yaml",
        ),
    )


def initialize_logging_outputs(*args: tuple, **kwargs: object) -> None:
    """Initialize the default logging console and file outputs.

    This function initializes the logging outputs based on configured
    parameters. Additional logging outputs may be provided.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Construct the default console and file-based logging functions. The file
    # is saved in the package directory.
    lezargus.library.logging.add_console_logging_handler(
        console=sys.stderr,
        log_level=lezargus.library.logging.LOGGING_INFO_LEVEL,
        use_color=lezargus.library.config.LOGGING_STREAM_USE_COLOR,
    )
    # The default file logging is really a temporary thing (just in case) and
    # should not kept from run to run. Moreover, if there are multiple
    # instances of Lezargus being run, they all cannot use the same log file
    # name and so we encode a UUID tag.

    # Adding a new file handler. We add the file handler first only so we can
    # capture the log messages when we try and remove the old logs.
    unique_hex_identifier = uuid.uuid4().hex
    default_log_file_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.library.config.INTERNAL_MODULE_INSTALLATION_PATH,
        filename="lezargus_" + unique_hex_identifier,
        extension="log",
    )
    lezargus.library.logging.add_file_logging_handler(
        filename=default_log_file_filename,
        log_level=lezargus.library.logging.LOGGING_DEBUG_LEVEL,
    )
    # We try and remove all of the log files which currently exist, if we can.
    # We make an exception for the one which we are going to use, we do not
    # want to clog the log with it.
    old_log_files = glob.glob(
        lezargus.library.path.merge_pathname(
            directory=lezargus.library.config.INTERNAL_MODULE_INSTALLATION_PATH,
            filename="lezargus_*",
            extension="log",
        ),
        recursive=False,
    )
    for filedex in old_log_files:
        if filedex == default_log_file_filename:
            # We do not try to delete the current file.
            continue
        try:
            os.remove(filedex)
        except OSError:
            # The file is likely in use by another logger or Lezargus instance.
            # The deletion can wait.
            logging.info(
                message=(
                    f"The temporary log file {filedex} is currently in-use, we"
                    " defer  deletion until the next load."
                ),
            )


def initialize_temporary_directory(*args: tuple, **kwargs: object) -> None:
    """Initialize the temporary directory.

    We create the temporary directory based on the configured paths.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # We need to get the temporary directory path, if the configurations were
    # not loaded, we inform the user.
    try:
        temporary_directory = (
            lezargus.library.config.LEZARGUS_TEMPORARY_DIRECTORY
        )
        # We also check for the flag filename because the creation of the the
        # directory includes it.
        temporary_flag_file = (
            lezargus.library.config.LEZARGUS_TEMPORARY_DIRECTORY_FLAG_FILENAME
        )
        overwrite = (
            lezargus.library.config.LEZARGUS_TEMPORARY_OVERWRITE_DIRECTORY
        )
    except AttributeError:
        # The configurations were likely not found.
        logging.critical(
            critical_type=logging.WrongOrderError,
            message=(
                "Configuration not initialized, temporary directory known."
                " Initialize the configurations or make the directory using"
                " library functions."
            ),
        )
    else:
        # We make the files.
        lezargus.library.temporary.create_temporary_directory(
            directory=temporary_directory,
            flag_filename=temporary_flag_file,
            overwrite=overwrite,
        )


def initialize_data_all(*args: tuple, **kwargs: object) -> None:
    """Initialize the all of the data files.

    Load all data files into the library data module.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Loading all of the data files.
    initialize_data_star_files(**kwargs)
    initialize_data_filter_files(**kwargs)
    initialize_data_atmosphere_files(**kwargs)

    # Computing the other data values.
    initialize_data_filter_zero_point_values(**kwargs)

    # All done.


def initialize_data_star_files(*args: tuple, **kwargs: object) -> None:
    """Initialize the stellar spectrum data files.

    Load all of stellar spectrum and other data files into the library data
    module.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Loading the stars.
    lezargus.library.data.add_data_object(
        name="STAR_16CYGB",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="star_spectra_16CygB",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="STAR_109VIR",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="star_spectra_109Vir",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="STAR_SUN",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="star_spectra_Sun",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="STAR_VEGA",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="star_spectra_Vega",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="STAR_A0V",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="star_spectra_A0V",
                extension="fits",
            ),
        ),
    )


def initialize_data_filter_files(*args: tuple, **kwargs: object) -> None:
    """Initialize the photometric filter data files.

    Load all of photometric filter data files into the library data module.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Loading the photometric filter files.

    # Loading Johnson filters.
    lezargus.library.data.add_data_object(
        name="FILTER_JOHNSON_U_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_U_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_JOHNSON_B_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_B_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_JOHNSON_V_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_V_photon",
                extension="fits",
            ),
        ),
    )

    # Loading GAIA filters.
    lezargus.library.data.add_data_object(
        name="FILTER_GAIA_GG_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GG_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_GAIA_GB_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GB_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_GAIA_GR_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GR_photon",
                extension="fits",
            ),
        ),
    )

    # Loading 2MASS filters.
    lezargus.library.data.add_data_object(
        name="FILTER_2MASS_J_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_J_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_2MASS_H_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_H_photon",
                extension="fits",
            ),
        ),
    )
    lezargus.library.data.add_data_object(
        name="FILTER_2MASS_KS_PHOTON",
        data=lezargus.container.LezargusSpectrum.read_fits_file(
            filename=lezargus.library.path.merge_pathname(
                directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_Ks_photon",
                extension="fits",
            ),
        ),
    )


def initialize_data_atmosphere_files(*args: tuple, **kwargs: object) -> None:
    """Initialize the PSG atmospheric data files.

    Load all of atmospheric transmission and emission data files into the
    library data module. We produce an atmospheric spectrum generator.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # The PSG atmospheric files are generated outside of this package and so
    # the defined zenith angles and precipitable water vapor values are known
    # before hand. The values here are defined based on the filenames and
    # should be valid for both transmission and radiance.
    zenith_angle_domain = np.array([0, 30, 45, 60])
    pwv_domain = np.array([0.5, 1.0, 2.0, 3.0])

    # They have a common estimated spectral resolution.
    psg_spectral_scale = 1e-9

    # First, transmission. We load the data produced.
    transmission_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
        filename="psg_telluric_transmission",
        extension="dat",
    )
    transmission_table = astropy.table.Table.read(
        transmission_filename,
        format="ascii.mrt",
    )
    # The domain is the zenith angles, PWV, and wavelength. The filenames use
    # angular degrees while the generator uses radians.
    transmission_wavelength = transmission_table["wavelength"]
    transmission_wavelength_unit = "m"
    zenith_angle_radians = np.deg2rad(zenith_angle_domain)
    # We package the transmission data so that it matches what the generator
    # expects.
    transmission_shape = (
        transmission_wavelength.size,
        len(zenith_angle_domain),
        len(pwv_domain),
    )
    transmission_array = np.empty(transmission_shape)
    transmission_array_unit = ""
    for zindex, zenithdex in enumerate(zenith_angle_domain):
        for pindex, pwvdex in enumerate(pwv_domain):
            column_name = f"za{zenithdex}_pwv{pwvdex}"
            transmission_array[:, zindex, pindex] = transmission_table[
                column_name
            ]
    # Creating the atmospheric transmission generator. We then add it to the
    # data module.
    transmission_generator = lezargus.container.AtmosphereSpectrumGenerator(
        wavelength=transmission_wavelength,
        zenith_angle=zenith_angle_radians,
        pwv=pwv_domain,
        data=transmission_array,
        wavelength_unit=transmission_wavelength_unit,
        data_unit=transmission_array_unit,
        spectral_scale=psg_spectral_scale,
    )
    lezargus.library.data.add_data_object(
        name="ATM_TRANS_GEN",
        data=transmission_generator,
    )

    # Second, we repeat for radiance. We load the data produced.
    radiance_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.library.config.INTERNAL_MODULE_DATA_DIRECTORY,
        filename="psg_telluric_radiance",
        extension="dat",
    )
    radiance_table = astropy.table.Table.read(
        radiance_filename,
        format="ascii.mrt",
    )
    # The domain is the zenith angles, PWV, and wavelength. We reuse the
    # zenith angle variable.
    radiance_wavelength = radiance_table["wavelength"]
    radiance_wavelength_unit = "m"
    # We package the radiance data so that it matches what the generator
    # expects.
    radiance_shape = (
        radiance_wavelength.size,
        len(zenith_angle_domain),
        len(pwv_domain),
    )
    radiance_array = np.empty(radiance_shape)
    radiance_array_unit = "W m^-2 sr^-1 m^-1"
    for zindex, zenithdex in enumerate(zenith_angle_domain):
        for pindex, pwvdex in enumerate(pwv_domain):
            column_name = f"za{zenithdex}_pwv{pwvdex}"
            radiance_array[:, zindex, pindex] = radiance_table[column_name]
    # Creating the atmospheric radiance generator. We then add it to the
    # data module.
    radiance_generator = lezargus.container.AtmosphereSpectrumGenerator(
        wavelength=radiance_wavelength,
        zenith_angle=zenith_angle_radians,
        pwv=pwv_domain,
        data=radiance_array,
        wavelength_unit=radiance_wavelength_unit,
        data_unit=radiance_array_unit,
        spectral_scale=psg_spectral_scale,
    )
    lezargus.library.data.add_data_object(
        name="ATM_RADIANCE_GEN",
        data=radiance_generator,
    )


def initialize_data_filter_zero_point_values(
    *args: tuple,
    **kwargs: object,
) -> None:
    """Initialize the PSG atmospheric data files.

    Load all of atmospheric transmission and emission data files into the
    library data module.

    Parameters
    ----------
    *args : tuple
        Positional arguments. There should be no positional arguments. This
        serves to catch them.
    **kwargs : dict
        A catch-all keyword argument, used to catch arguments which are not
        relevant or are otherwise passed to other internal functions.

    Returns
    -------
    None

    """
    # The initialization function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Initialization cannot have positional arguments, use keyword"
                " arguments."
            ),
        )
    # This is to "use" the kwarg parameter, nothing much else.
    lezargus.library.wrapper.do_nothing(**kwargs)

    # Calculating Johnson filters zero point values.
    # Johnson U band.
    (
        johnson_u_zp,
        johnson_u_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_JOHNSON_U_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_J_U"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_J_U"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_U",
        data=johnson_u_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_U_UNCERTAINTY",
        data=johnson_u_zpu,
    )
    # Johnson B band.
    (
        johnson_b_zp,
        johnson_b_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_JOHNSON_B_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_J_B"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_J_B"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_B",
        data=johnson_b_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_B_UNCERTAINTY",
        data=johnson_b_zpu,
    )
    # Johnson V band.
    (
        johnson_v_zp,
        johnson_v_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_JOHNSON_V_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_J_V"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_J_V"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_V",
        data=johnson_v_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_V_UNCERTAINTY",
        data=johnson_v_zpu,
    )

    # Calculating Gaia filters zero point values.
    logging.error(
        error_type=logging.ToDoError,
        message="Gaia zero point filter values need to be calculated.",
    )

    # Calculating 2MASS filters zero point values.
    # 2MASS J band.
    (
        mass2_j_zp,
        mass2_j_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_2MASS_J_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_2_J"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_2_J"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_J",
        data=mass2_j_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_J_UNCERTAINTY",
        data=mass2_j_zpu,
    )
    # 2MASS H band.
    (
        mass2_h_zp,
        mass2_h_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_2MASS_H_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_2_H"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_2_H"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_H",
        data=mass2_h_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_H_UNCERTAINTY",
        data=mass2_h_zpu,
    )
    # 2MASS Ks band.
    (
        mass2_ks_zp,
        mass2_ks_zpu,
    ) = lezargus.library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=lezargus.library.data.FILTER_2MASS_KS_PHOTON,
        standard_spectra=lezargus.library.data.STAR_A0V,
        standard_filter_magnitude=lezargus.library.data.STAR_A0V.header[
            "LZPM_2KS"
        ],
        standard_filter_uncertainty=lezargus.library.data.STAR_A0V.header[
            "LZPU_2KS"
        ],
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_KS",
        data=mass2_ks_zp,
    )
    lezargus.library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_KS_UNCERTAINTY",
        data=mass2_ks_zpu,
    )

    # All done.
