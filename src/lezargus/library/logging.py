"""Error, warning, and logging functionality pertinent to Lezargus.

Use the functions here when logging or issuing errors or other information.
"""

import logging
import string

import colorama

from lezargus import library
from lezargus.library import hint


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


class LezargusError(Exception):
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


class DirectoryError(LezargusError):
    """An error used for directory issues.

    If there are issues with directories, use this error.
    """


class ElevatedError(LezargusError):
    """An error used when elevating warnings or errors to critical level.

    Only to be used when elevating via the configuration property.
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


class ConfigurationWarning(LezargusWarning):
    """A warning for inappropriate configurations.

    This warning is to be used when the configuration file is wrong. There is a
    specific expectation for how configuration files and configuration
    parameters are structures are defined.
    """


class InputWarning(LezargusWarning):
    """A warning for a weird input.

    This warning is used when the input of a function or a field is not
    expected, but may be able to be handled.
    """


###############################################################################

# Logging levels alias.
LOGGING_DEBUG_LEVEL = logging.DEBUG
LOGGING_INFO_LEVEL = logging.INFO
LOGGING_WARNING_LEVEL = logging.WARNING
LOGGING_ERROR_LEVEL = logging.ERROR
LOGGING_CRITICAL_LEVEL = logging.CRITICAL
# The logger itself.
__lezargus_logger = logging.getLogger()
__lezargus_logger.setLevel(LOGGING_DEBUG_LEVEL)


class ColoredLogFormatter(logging.Formatter):
    """Use this formatter to have colors.

    Attributes
    ----------
    message_format : str
        The message format, passed directly to the logger formatter after
        the color keys are added.
    date_format : str
        The date format, passed directly to the logger formatter.
    color_formatting : dict
        The formatting for the color.
    """

    def __init__(
        self: "ColoredLogFormatter",
        message_format: str,
        date_format: str,
        color_hex_dict: dict[int:str] = None,
    ) -> None:
        """Initialize the color formatter.

        Parameters
        ----------
        message_format : str
            The message format, passed directly to the logger formatter after
            the color keys are added.
        date_format : str
            The date format, passed directly to the logger formatter.
        color_hex_dict : dict, default = None
            The dictionary containing the color pairings between logging
            levels and its actual color. It should be a
            {level_number:hex_color} dictionary.

        Returns
        -------
        None
        """
        super().__init__()
        # The default.
        color_hex_dict = {} if color_hex_dict is None else color_hex_dict
        # Get the escape codes from the HEX colors.
        self.message_format = message_format
        self.date_format = date_format
        reset_ansi_escape = "\x1b[0m"
        # Establishing the formatting.
        self.color_formatting = {}
        for leveldex, colordex in color_hex_dict.items():
            # Skip cases where there is no color.
            if len(colordex) == 0:
                continue
            color_ansi_escape = self.__convert_color_hex_to_ansi_escape(
                color_hex=colordex,
            )
            self.color_formatting[leveldex] = (
                color_ansi_escape + self.message_format + reset_ansi_escape
            )
        # Flip the Windows color flag compatibility.
        colorama.just_fix_windows_console()

    def format(  # noqa: A003
        self: "ColoredLogFormatter",
        record: hint.LogRecord,
    ) -> str:
        """Format a log record.

        The name of this function cannot be helped as it is required for the
        Python logging module.

        Parameters
        ----------
        record : LogRecord
            The record to format.

        Returns
        -------
        formatted_record : str
            The formatted string.
        """
        log_format = self.color_formatting.get(record.levelno, None)
        formatter = logging.Formatter(fmt=log_format, datefmt=self.date_format)
        return formatter.format(record)

    @staticmethod
    def __convert_color_hex_to_ansi_escape(color_hex: str) -> str:
        """Convert a hex code to a ANSI escape code.

        Parameters
        ----------
        color_hex : str
            The HEX code of the color, including the # symbol.

        Returns
        -------
        color_ansi_escape : str
            The ANSI escape code for the color.
        """
        # If the color code is not a hex, give a warning.
        color_hex = color_hex.upper()
        # Does it start with a hash?
        hash_check = color_hex[0] == "#"
        # Is it only 6 characters and the hash?
        hex_length = 7
        length_check = len(color_hex) == hex_length
        # Does it contain only HEX digits?
        char_check = all(
            chardex in set(string.hexdigits) for chardex in color_hex[1:]
        )
        if not (hash_check and length_check and char_check):
            warning(
                warning_type=InputWarning,
                message=(
                    "The following HEX color code input is not a proper HEX"
                    " color code: {hx}".format(hx=color_hex)
                ),
            )
        # Converting from HEX string to the RGB color code.
        color_ansi_escape = "\033[38;2;{r};{g};{b}m".format(
            r=int(color_hex[1:3], 16),
            g=int(color_hex[3:5], 16),
            b=int(color_hex[5:7], 16),
        )
        return color_ansi_escape


def add_stream_logging_handler(
    stream: object,
    log_level: int = LOGGING_DEBUG_LEVEL,
    use_color: bool = True,
) -> None:
    """Add a stream handler to the logging infrastructure.

    Parameters
    ----------
    stream : Any
        The stream where the logs will write to.
    log_level : int
        The logging level for this handler.
    use_color : bool
        If True, use colored log messaged based on the configuration file
        parameters.

    Returns
    -------
    None
    """
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(log_level)
    # Get the format from the specified configuration.
    color_format_dict = {
        LOGGING_DEBUG_LEVEL: library.config.LOGGING_STREAM_DEBUG_COLOR_HEX,
        LOGGING_INFO_LEVEL: library.config.LOGGING_STREAM_INFO_COLOR_HEX,
        LOGGING_WARNING_LEVEL: library.config.LOGGING_STREAM_WARNING_COLOR_HEX,
        LOGGING_ERROR_LEVEL: library.config.LOGGING_STREAM_ERROR_COLOR_HEX,
        LOGGING_CRITICAL_LEVEL: (
            library.config.LOGGING_STREAM_CRITICAL_COLOR_HEX
        ),
    }
    if use_color:
        stream_formatter = ColoredLogFormatter(
            message_format=library.config.LOGGING_RECORD_FORMAT_STRING,
            date_format=library.config.LOGGING_DATETIME_FORMAT_STRING,
            color_hex_dict=color_format_dict,
        )
    else:
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
    log_level: int = LOGGING_DEBUG_LEVEL,
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


def update_global_minimum_logging_level(
    log_level: int = LOGGING_DEBUG_LEVEL,
) -> None:
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


def warning(
    warning_type: LezargusWarning,
    message: str,
    elevate: bool = None,
) -> None:
    """Log a warning message.

    This is a wrapper around the warning function to standardize it for
    Lezargus.

    Parameters
    ----------
    warning_type : LezargusWarning
        The class of the warning which will be used.
    message : str
        The warning message.
    elevate : bool, default = None
        If True, always elevate the warning to a critical issue. By default,
        use the configuration value.

    Returns
    -------
    None
    """
    # Check if the warning type provided is a Lezargus type.
    if not issubclass(warning_type, LezargusWarning):
        critical(
            critical_type=DevelopmentError,
            message=(
                "The provided warning type `{ty}` is not a subclass of the"
                " Lezargus warning type.".format(ty=warning_type)
            ),
        )
    # We add the warning type to the message, if the configuration specifies it
    # to be so.
    if library.config.LOGGING_INCLUDE_EXCEPTION_TYPE_IN_MESSAGE:
        typed_message = f"{warning_type.__name__} - {message}"
    else:
        # Do not add anything.
        typed_message = message

    # Now we issue the warning.
    __lezargus_logger.warning(typed_message)

    # If the warning should be elevated.
    elevate = (
        elevate
        if elevate is not None
        else library.config.LOGGING_ELEVATE_WARNING_TO_CRITICAL
    )
    if elevate:
        elevated_message = (
            f"The following warning was elevated: {typed_message}"
        )
        critical(critical_type=ElevatedError, message=elevated_message)


def error(
    error_type: LezargusError,
    message: str,
    elevate: bool = None,
) -> None:
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
    elevate : bool, default = None
        If True, always elevate the error to a critical issue. By default,
        use the configuration value.

    Returns
    -------
    None
    """
    # Check if the warning type provided is a Lezargus type.
    if not issubclass(error_type, LezargusError | LezargusBaseError):
        critical(
            critical_type=DevelopmentError,
            message=(
                "The provided error type `{ty}` is not a subclass of the"
                " Lezargus error type.".format(ty=error_type)
            ),
        )
    # We add the error type to the message, if the configuration specifies it
    # to be so.
    if library.config.LOGGING_INCLUDE_EXCEPTION_TYPE_IN_MESSAGE:
        typed_message = f"{error_type.__name__} - {message}"
    else:
        # Do not add anything.
        typed_message = message
    # The error type needs to be something that can be used for warning.
    # Typically, only the non-base errors will be used anyways.
    __lezargus_logger.error(typed_message)
    # If the error should be elevated.
    elevate = (
        elevate
        if elevate is not None
        else library.config.LOGGING_ELEVATE_ERROR_TO_CRITICAL
    )
    if elevate:
        elevated_message = f"The following error was elevated: {typed_message}"
        critical(critical_type=ElevatedError, message=elevated_message)


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
                "The provided critical type `{ty}` is not a subclass of the"
                " Lezargus error type.".format(ty=critical_type)
            ),
        )
    # We add the critical type to the message, if the configuration specifies it
    # to be so.
    if library.config.LOGGING_INCLUDE_EXCEPTION_TYPE_IN_MESSAGE:
        typed_message = f"{critical_type.__name__} - {message}"
    else:
        # Do not add anything.
        typed_message = message

    __lezargus_logger.critical(typed_message)
    # Finally, we raise/throw the error.
    raise critical_type(message)


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
