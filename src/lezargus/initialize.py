"""Module, file, and data initialization routines of Lezargus.

Everything and anything which initializes Lezargus, that is separate from
Python loading this module, is done here.
"""

import glob
import os
import sys
import uuid

import lezargus
from lezargus import library
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to full initialization present."
            ),
        )

    # Load in the default configuration file.
    initialize_configuration(**kwargs)

    # Load the logging outputs.
    initialize_logging_outputs(**kwargs)

    # All of the initializations below have logging.

    # Load all of the data files for Lezargus.
    initialize_data_all_files(**kwargs)


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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to configuration initialization"
                " present."
            ),
        )

    # Load the default configuration parameters. The user's configurations
    # should overwrite these when supplied.
    library.config.load_then_apply_configuration(
        filename=library.path.merge_pathname(
            directory=library.config.MODULE_INSTALLATION_PATH,
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to logging initialization"
                " present."
            ),
        )

    # Construct the default console and file-based logging functions. The file
    # is saved in the package directory.
    library.logging.add_console_logging_handler(
        console=sys.stderr,
        log_level=library.logging.LOGGING_INFO_LEVEL,
        use_color=library.config.LOGGING_STREAM_USE_COLOR,
    )
    # The default file logging is really a temporary thing (just in case) and
    # should not kept from run to run. Moreover, if there are multiple
    # instances of Lezargus being run, they all cannot use the same log file
    # name and so we encode a UUID tag.

    # Adding a new file handler. We add the file handler first only so we can
    # capture the log messages when we try and remove the old logs.
    unique_hex_identifier = uuid.uuid4().hex
    default_log_file_filename = library.path.merge_pathname(
        directory=library.config.MODULE_INSTALLATION_PATH,
        filename="lezargus_" + unique_hex_identifier,
        extension="log",
    )
    library.logging.add_file_logging_handler(
        filename=default_log_file_filename,
        log_level=library.logging.LOGGING_DEBUG_LEVEL,
    )
    # We try and remove all of the log files which currently exist, if we can.
    # We make an exception for the one which we are going to use, we do not
    # want to clog the log with it.
    old_log_files = glob.glob(
        library.path.merge_pathname(
            directory=library.config.MODULE_INSTALLATION_PATH,
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
            library.logging.info(
                message=(
                    f"The temporary log file {filedex} is currently in-use, we"
                    " defer  deletion until the next load."
                ),
            )


def initialize_data_all_files(*args: tuple, **kwargs: object) -> None:
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to data file initialization"
                " present."
            ),
        )

    # Loading all of the data files.
    initialize_data_star_files(**kwargs)
    initialize_data_filter_files(**kwargs)

    # Computing the other data values.
    initialize_data_filter_zero_point_values(**kwargs)

    # All done.


def initialize_data_star_files(*args: tuple, **kwargs: object) -> None:
    """Initialize the stellar spectra data files.

    Load all of stellar spectra and other data files into the library data
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to data star file initialization"
                " present."
            ),
        )

    # Loading the stars.
    library.data.add_data_object(
        name="STAR_16CYGB",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="star_spectra_16CygB",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="STAR_109VIR",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="star_spectra_109Vir",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="STAR_SUN",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="star_spectra_Sun",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="STAR_VEGA",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="star_spectra_Vega",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="STAR_A0V",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to data photometric filter file"
                " initialization present."
            ),
        )

    # Loading the photometric filter files.

    # Loading Johnson filters.
    library.data.add_data_object(
        name="FILTER_JOHNSON_U_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_U_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_JOHNSON_B_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_B_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_JOHNSON_V_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Johnson_V_photon",
                extension="fits",
            ),
        ),
    )

    # Loading GAIA filters.
    library.data.add_data_object(
        name="FILTER_GAIA_GG_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GG_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_GAIA_GB_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GB_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_GAIA_GR_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_Gaia_GR_photon",
                extension="fits",
            ),
        ),
    )

    # Loading 2MASS filters.
    library.data.add_data_object(
        name="FILTER_2MASS_J_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_J_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_2MASS_H_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_H_photon",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="FILTER_2MASS_KS_PHOTON",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="filter_2MASS_Ks_photon",
                extension="fits",
            ),
        ),
    )


def initialize_data_atmosphere_files(*args: tuple, **kwargs: object) -> None:
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to data atmospheric file"
                " initialization present."
            ),
        )

    # We first load all of the atmospheric transmission files.
    library.data.add_data_object(
        name="ATM_TRANS_ZA0",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_trans_za0",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_TRANS_ZA30",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_trans_za30",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_TRANS_ZA45",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_trans_za45",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_TRANS_ZA60",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_trans_za60",
                extension="fits",
            ),
        ),
    )

    # We next load all of the atmospheric radiance files.
    library.data.add_data_object(
        name="ATM_RAD_ZA0",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_rad_za0",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_RAD_ZA30",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_rad_za30",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_RAD_ZA45",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_rad_za45",
                extension="fits",
            ),
        ),
    )
    library.data.add_data_object(
        name="ATM_RAD_ZA60",
        data=lezargus.container.LezargusSpectra.read_fits_file(
            filename=library.path.merge_pathname(
                directory=library.data.MODULE_DATA_DIRECTORY,
                filename="atm_rad_za60",
                extension="fits",
            ),
        ),
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
    if len(kwargs) != 0:
        logging.debug(
            message=(
                "Overriding keyword parameters to data zero point value"
                " initialization present."
            ),
        )

    # Calculating Johnson filters zero point values.
    # Johnson U band.
    (
        johnson_u_zp,
        johnson_u_zpu,
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_JOHNSON_U_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_J_U"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_J_U"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_U",
        data=johnson_u_zp,
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_U_UNCERTAINTY",
        data=johnson_u_zpu,
    )
    # Johnson B band.
    (
        johnson_b_zp,
        johnson_b_zpu,
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_JOHNSON_B_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_J_B"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_J_B"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_B",
        data=johnson_b_zp,
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_B_UNCERTAINTY",
        data=johnson_b_zpu,
    )
    # Johnson V band.
    (
        johnson_v_zp,
        johnson_v_zpu,
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_JOHNSON_V_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_J_V"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_J_V"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_JOHNSON_V",
        data=johnson_v_zp,
    )
    library.data.add_data_object(
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
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_2MASS_J_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_2_J"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_2_J"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_J",
        data=mass2_j_zp,
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_J_UNCERTAINTY",
        data=mass2_j_zpu,
    )
    # 2MASS H band.
    (
        mass2_h_zp,
        mass2_h_zpu,
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_2MASS_H_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_2_H"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_2_H"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_H",
        data=mass2_h_zp,
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_H_UNCERTAINTY",
        data=mass2_h_zpu,
    )
    # 2MASS Ks band.
    (
        mass2_ks_zp,
        mass2_ks_zpu,
    ) = library.photometry.calculate_filter_zero_point_vega(
        filter_spectra=library.data.FILTER_2MASS_KS_PHOTON,
        standard_spectra=library.data.STAR_A0V,
        standard_filter_magnitude=library.data.STAR_A0V.header["LZPM_2KS"],
        standard_filter_uncertainty=library.data.STAR_A0V.header["LZPU_2KS"],
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_KS",
        data=mass2_ks_zp,
    )
    library.data.add_data_object(
        name="ZERO_POINT_VEGA_2MASS_KS_UNCERTAINTY",
        data=mass2_ks_zpu,
    )

    # All done.
