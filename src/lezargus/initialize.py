"""Module, file, and data initialization routines of Lezargus.

Everything and anything which initializes Lezargus, that is separate from
Python loading this module, is done here. After the program is run, we clean
up using terminate.py.
"""

import glob
import os
import sys
import uuid

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

    # Load the logging outputs.
    initialize_logging_outputs(**kwargs)

    # All of the initializations below have logging.

    # Loading and creating the needed temporary directories.
    initialize_temporary_directory(**kwargs)


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
        use_color=lezargus.config.LOGGING_STREAM_USE_COLOR,
    )
    # The default file logging is really a temporary thing (just in case) and
    # should not kept from run to run. Moreover, if there are multiple
    # instances of Lezargus being run, they all cannot use the same log file
    # name and so we encode a UUID tag.

    # Adding a new file handler. We add the file handler first only so we can
    # capture the log messages when we try and remove the old logs.
    unique_hex_identifier = uuid.uuid4().hex
    default_log_file_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.config.INTERNAL_MODULE_INSTALLATION_PATH,
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
            directory=lezargus.config.INTERNAL_MODULE_INSTALLATION_PATH,
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
            logging.debug(
                message=(
                    f"The temporary log file {filedex} is currently in-use, we"
                    " defer deletion until the next load."
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
    temporary_directory = lezargus.config.LEZARGUS_TEMPORARY_DIRECTORY
    # We also check for the flag filename because the creation of the the
    # directory includes it.
    temporary_flag_file = (
        lezargus.config.LEZARGUS_TEMPORARY_DIRECTORY_FLAG_FILENAME
    )
    overwrite = lezargus.config.LEZARGUS_TEMPORARY_OVERWRITE_DIRECTORY

    # We make the files.
    lezargus.library.temporary.create_temporary_directory(
        directory=temporary_directory,
        flag_filename=temporary_flag_file,
        overwrite=overwrite,
    )
