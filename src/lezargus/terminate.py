"""Module, file, and data termination routines of Lezargus.

Everything and anything which terminates Lezargus is done here. We do the
opposite of initialize.py, cleaning up after ourselves.
"""

import lezargus
from lezargus.library import hint
from lezargus.library import logging


def terminate(*args: tuple, **kwargs: object) -> None:
    """Terminate the Lezargus module and all its parts, cleaning up.

    The termination function, breaking down what we created and initialized
    and cleaning up after ourselves.

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

    # Unload all of the data files for Lezargus.
    # - All of the data files are cleaned up on their own.

    # Remove any temporary directory and files, if desired.
    terminate_temporary_directory(**kwargs)

    # Remove all of the logging outputs.
    # - All of the logging is cleaned up on its own.


def terminate_temporary_directory(*args: tuple, **kwargs: hint.Any) -> None:
    """Terminate the temporary directory.

    We remove the temporary directory based on the configured paths.

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
    # The termination function cannot have positional arguments as
    # such positional arguments may get confused for other arguments when
    # we pass it down.
    if len(args) != 0:
        logging.critical(
            critical_type=logging.InputError,
            message=(
                "Termination cannot have positional arguments, use keyword"
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
    force = lezargus.config.LEZARGUS_TEMPORARY_FORCE_DELETION

    # We remove the files, if the user wants to of course.
    if not lezargus.config.LEZARGUS_TEMPORARY_SKIP_DELETION:
        lezargus.library.temporary.delete_temporary_directory(
            directory=temporary_directory,
            flag_filename=temporary_flag_file,
            force=force,
        )
