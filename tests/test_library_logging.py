"""This file contains tests that test the Lezargus logging and error code."""

import lezargus


def test_update_global_minimum_logging_level() -> None:
    """Test the update_global_minimum_logging_level function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We just run the logging again to test it.
    __ = lezargus.library.logging.update_global_minimum_logging_level(
        log_level=lezargus.library.logging.LOGGING_DEBUG_LEVEL,
    )


def test_debug() -> None:
    """Test the debug function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    lezargus.library.logging.debug(message="Debug test.")


def test_info() -> None:
    """Test the info function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    lezargus.library.logging.info(message="Info test.")


def test_warning() -> None:
    """Test the warning function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We try to send a warning with a non-Lezargus warning type.
    try:
        lezargus.library.logging.warning(
            warning_type=lezargus.library.logging.LezargusError,
            message="Using an error for a warning.",
        )
    except lezargus.library.logging.DevelopmentError:
        # The error is expected.
        pass

    # We send a normal warning.
    lezargus.library.logging.warning(
        warning_type=lezargus.library.logging.LezargusWarning,
        message="Warning test.",
        elevate=False,
    )

    # We now send a warning which is to be elevated.
    try:
        lezargus.library.logging.warning(
            warning_type=lezargus.library.logging.LezargusWarning,
            message="Elevate warning test.",
            elevate=True,
        )
    except lezargus.library.logging.ElevatedError:
        # The error is expected.
        pass


def test_error() -> None:
    """Test the error function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We try to send a warning with a non-Lezargus error type.
    try:
        lezargus.library.logging.error(
            error_type=lezargus.library.logging.LezargusWarning,
            message="Using a warning for an error.",
        )
    except lezargus.library.logging.DevelopmentError:
        # The error is expected.
        pass

    # We send a normal warning.
    lezargus.library.logging.error(
        error_type=lezargus.library.logging.LezargusError,
        message="Error test.",
        elevate=False,
    )

    # We now send a warning which is to be elevated.
    try:
        lezargus.library.logging.error(
            error_type=lezargus.library.logging.LezargusError,
            message="Elevate error test.",
            elevate=True,
        )
    except lezargus.library.logging.ElevatedError:
        # The error is expected.
        pass


def test_critical() -> None:
    """Test the critical function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    # We try to send a warning with a non-Lezargus error/critical type.
    try:
        lezargus.library.logging.critical(
            critical_type=lezargus.library.logging.LezargusWarning,
            message="Using a warning for an error.",
        )
    except lezargus.library.logging.DevelopmentError:
        # The error is expected.
        pass

    # We send a normal critical.
    try:
        lezargus.library.logging.critical(
            critical_type=lezargus.library.logging.UndiscoveredError,
            message="Error test.",
        )
    except lezargus.library.logging.UndiscoveredError:
        # The error is expected.
        pass


def test_terminal() -> None:
    """Test the terminal function.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    try:
        lezargus.library.logging.terminal()
    except lezargus.library.logging.LezargusBaseError:
        # The error is expected.
        pass
