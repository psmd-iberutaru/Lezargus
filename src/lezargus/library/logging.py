"""Error, warning, and logging functionality pertinent to Lezargus.

Use the functions here when logging or issuing errors or other information.
"""

import logging
import warnings

from lezargus import library


class LezargusBaseError(BaseException):
    """The base inheriting class which for all Lezargus errors.

    This is for exceptions that should never be caught and should bring
    everything to a halt.
    """


class DevelopmentError(LezargusBaseError):
    """An error used for a development error.

    This is an error where the development of Lezargus is not correct and
    something is not coded based on the expectations of the software itself.
    This is not the fault of the user.
    """


class LogicFlowError(LezargusBaseError):
    """An error used for an error in the flow of program logic.

    This is an error to ensure that the logic does not flow to a point to a
    place where it is not supposed to. This is helpful in making sure changes
    to the code do not screw up the logical flow of the program.
    """


class BeyondScopeError(LezargusBaseError):
    """An error used for something which is beyond the scope of work.

    This is an error to be used when what is trying to be done does not
    seem reasonable. Usually warnings are the better thing for this but
    this error is used when the assumptions for reasonability guided
    development and what the user is trying to do is not currently supported
    by the software.
    """


class UndiscoveredError(LezargusBaseError):
    """An error used for an unknown error.

    This is an error used in cases where the source of the error has not
    been determined and so a more helpful error message or mitigation strategy
    cannot be devised.
    """


class LezargusError(UserWarning, Exception):
    """The main inheriting class which all Lezargus errors use as their base.

    This is done for ease of error handling and is something that can and
    should be managed.
    """


class CommandLineError(LezargusError):
    """An error used for an error with the command-line.

    This error is used when the entered command-line command or its options
    are not correct.
    """


class ConfigurationError(LezargusError):
    """An error used for an error with the configuration file.

    This error is to be used when the configuration file is wrong. There is a
    specific expectation for how configuration files and configuration
    parameters are structures are defined.
    """


class CriticalError(LezargusError):
    """An error used for a critical error.

    This is a wrapper error class which only exists to better integrate
    with the logger/warning hybrid message system.
    """


class DirectoryError(LezargusError):
    """An error used for directory issues.

    If there are issues with directories, use this error.
    """


class FileError(LezargusError):
    """An error used for file issues.

    If there are issues with files, use this error. This error should not be
    used in cases where the problem is because of an incorrect format of the
    file (other than corruption).
    """


class InputError(LezargusError):
    """An error used for issues with input parameters or data.

    This is the error to be used when the inputs to a function are not valid
    and do not match the expectations of that function.
    """


class ReadOnlyError(LezargusError):
    """An error used for problems with read-only files.

    If the file is read-only and it needs to be read, use FileError. This
    error is to be used only when variables or files are assumed to be read
    only, this error should be used to enforce that notion.
    """


class SequentialOrderError(LezargusError):
    """An error used when things are done out-of-order.

    This error is used when something is happening out of the expected required
    order. This order being in place for specific publicly communicated
    reasons.
    """


class LezargusWarning(UserWarning):
    """The main inheriting class which all Lezargus warnings use as their base.

    The base warning class which all of the other Lezargus warnings
    are derived from.
    """


class AccuracyWarning(LezargusWarning):
    """A warning for inaccurate results.

    This warning is used when some elements of the simulation or data
    reduction would yield less than desireable results.
    """


###############################################################################

__lezargus_logger = logging.getLogger()
__lezargus_logger.setLevel(logging.DEBUG)


def add_stream_logging_handler(
    stream: object,
    log_level: int = logging.DEBUG,
) -> None:
    """Add a stream handler to the logging infrastructure.

    Parameters
    ----------
    stream : Any
        The stream where the logs will write to.
    log_level : int
        The logging level for this handler.

    Returns
    -------
    None
    """
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(log_level)
    # Get the format from the specified configuration.
    stream_formatter = logging.Formatter(
        fmt=library.config.LOGGING_RECORD_FORMAT_STRING,
        datefmt=library.config.LOGGING_DATETIME_FORMAT_STRING,
    )
    # Adding the logger.
    stream_handler.setFormatter(stream_formatter)
    __lezargus_logger.addHandler(stream_handler)
    # All done.


def add_file_logging_handler(
    filename: str,
    log_level: int = logging.DEBUG,
) -> None:
    """Add a stream handler to the logging infrastructure.

    Parameters
    ----------
    filename : str
        The filename path where the log file will be saved to.
    log_level : int
        The logging level for this handler.

    Returns
    -------
    None
    """
    file_handler = logging.FileHandler(filename, "a")
    file_handler.setLevel(log_level)
    # Get the format from the specified configuration.
    file_formatter = logging.Formatter(
        fmt=library.config.LOGGING_RECORD_FORMAT_STRING,
        datefmt=library.config.LOGGING_DATETIME_FORMAT_STRING,
    )
    # Adding the logger.
    file_handler.setFormatter(file_formatter)
    __lezargus_logger.addHandler(file_handler)
    # All done.


def update_global_minimum_logging_level(log_level: int = logging.DEBUG) -> None:
    """Update the logging level of this module.

    This function updates the minimum logging level which is required for
    a log record to be recorded. Handling each single logger handler is really
    unnecessary.

    Parameters
    ----------
    log_level : int, default = logging.DEBUG
        The log level which will be set as the minimum level.

    Returns
    -------
    None
    """
    # Setting the log level.
    __lezargus_logger.setLevel(log_level)
    # ...and the level of the handlers.
    for handlerdex in __lezargus_logger.handlers:
        handlerdex.setLevel(log_level)
    # All done.


def debug(message: str) -> None:
    """Log a debug message.

    This is a wrapper around the debug function to standardize it for Lezargus.

    Parameters
    ----------
    message : str
        The debugging message.

    Returns
    -------
    None
    """
    __lezargus_logger.debug(message)


def info(message: str) -> None:
    """Log an informational message.

    This is a wrapper around the info function to standardize it for Lezargus.

    Parameters
    ----------
    message : str
        The informational message.

    Returns
    -------
    None
    """
    __lezargus_logger.info(message)


def warning(warning_type: LezargusWarning, message: str) -> None:
    """Log a warning message.

    This is a wrapper around the warning function to standardize it for
    Lezargus.

    Parameters
    ----------
    warning_type : LezargusWarning
        The class of the warning which will be used.
    message : str
        The warning message.

    Returns
    -------
    None
    """
    # Check if the warning type provided is a Lezargus type.
    if not issubclass(warning_type, LezargusWarning):
        critical(
            critical_type=DevelopmentError,
            message=(
                "The provided warning is not a subclass of the Lezargus warning"
                " type."
            ),
        )
    else:
        warnings.warn(message, warning_type, stacklevel=2)
        __lezargus_logger.warning(message)


def error(error_type: LezargusError, message: str) -> None:
    """Log an error message, do not raise.

    Use this for issues which are more serious than warnings but do not result
    in a raised exception.

    This is a wrapper around the error function to standardize it for Lezargus.

    Parameters
    ----------
    error_type : LezargusError
        The class of the error which will be used.
    message : str
        The error message.

    Returns
    -------
    None
    """
    # Check if the warning type provided is a Lezargus type.
    if not issubclass(error_type, LezargusError | LezargusBaseError):
        critical(
            critical_type=DevelopmentError,
            message=(
                "The provided error is not a subclass of the Lezargus error"
                " types."
            ),
        )
    else:
        # The error type needs to be something that can be used for warning.
        # Typically, only the non-base errors will be used anyways.
        error_type = (
            error_type
            if issubclass(error_type, LezargusError)
            else LezargusError
        )
        warnings.warn(message, error_type, stacklevel=2)
        __lezargus_logger.error(message)


def critical(critical_type: LezargusError, message: str) -> None:
    """Log a critical error and raise.

    Use this for issues which are more serious than warnings and should
    raise/throw an exception. The main difference between critical and error
    for logging is that critical will also raise the exception as error will
    not and the program will attempt to continue.

    This is a wrapper around the critical function to standardize it for
    Lezargus.

    Parameters
    ----------
    critical_type : LezargusError
        The class of the critical exception error which will be used and
        raised.
    message : str
        The critical error message.

    Returns
    -------
    None
    """
    # Check if the warning type provided is a Lezargus type.
    if not issubclass(critical_type, LezargusError | LezargusBaseError):
        critical(
            critical_type=DevelopmentError,
            message=(
                "The provided critical error is not a subclass of the Lezargus"
                " error types."
            ),
        )
    else:
        warning_message = "[{crit_type}] {msg}".format(
            crit_type=critical_type,
            msg=message,
        )
        warnings.warn(warning_message, CriticalError, stacklevel=2)
        __lezargus_logger.critical(message)
        # Finally, we raise/throw the error.
        raise critical_type(message)
    # The code should not get here.
    critical(
        critical_type=LogicFlowError,
        message="The code should not be here.",
    )


def terminal() -> None:
    """Terminal error function which is used to stop everything.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Raise.
    msg = (
        "TERMINAL - This is a general exception, see the traceback for more"
        " information."
    )
    raise LezargusBaseError(
        msg,
    )
