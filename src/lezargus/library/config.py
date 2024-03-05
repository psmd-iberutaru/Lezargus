"""Controls the inputting of configuration files.

This also serves to bring all of the configuration parameters into a more
accessible space which other parts of Lezargus can use.

Note these configuration constant parameters are all accessed using capital
letters regardless of the configuration file's labels. Because of this, the
names must also obey a stricter set of Python variable naming conventions.
Namely, capital letter names and only alphanumeric characters.

There are constant parameters which are stored here which are not otherwise
changeable by the configuration file.
"""

import os
import shutil

import yaml

import lezargus
from lezargus.library import logging


def sanitize_configuration(configuration: dict) -> dict:
    """Sanitize the configuration, conforming it to the Lezargus standards.

    Sometimes configurations input by users do not exactly follow the
    expectations of Lezargus, so, here, we sanitize it as much as we can.
    Should some level of sanitation fail, then we inform the user.

    Parameters
    ----------
    configuration : dict
        The configuration we are going to sanitize.

    Returns
    -------
    sanitized_configuration : dict
        The configuration, after sanitization.

    """
    # We need to entry by entry in sanitization.
    sanitized_configuration = {}
    for keydex, valuedex in configuration.items():
        # We first need to sanitize the key.
        sanitized_key = sanitize_configuration_key(key=keydex)
        # And the value...
        sanitized_value = sanitize_configuration_value(value=valuedex)
        # Reconstruction of the dictionary.
        sanitized_configuration[sanitized_key] = sanitized_value
    # All done.
    return sanitized_configuration


def sanitize_configuration_key(key: str) -> str:
    """Sanitize only the configuration key name.

    The key sanitization makes sure that the key follows the below criteria:

    - The key contains only letters and single underscores as word
      demarcations.
    - The key is all uppercase and is unique across all variations of cases.

    Parameters
    ----------
    key : str
        The configuration key to sanitize.

    Returns
    -------
    sanitized_key : str
        The key, sanitized.

    """
    # We replace common text demarcations with underscores. Also,
    # only single underscores so we need to remove subsequent underscores.
    common_demarcations = [" ", "-", "."]
    underscore_key = key
    for demarkdex in common_demarcations:
        underscore_key = underscore_key.replace(demarkdex, "_")
    has_successive_underscores = "__" in underscore_key
    while has_successive_underscores:
        # Underscore check.
        has_successive_underscores = "__" in underscore_key
        underscore_key = underscore_key.replace("__", "_")

    # We check that it only has letters.
    letter_test_key = underscore_key.replace("_", "")
    if not letter_test_key.isalnum():
        logging.critical(
            critical_type=logging.ConfigurationError,
            message=(
                f"Key {key} contains non-alphanumeric non-underscore"
                " characters."
            ),
        )
    if not (underscore_key[0].isascii() and underscore_key[0].isalpha()):
        logging.critical(
            critical_type=logging.ConfigurationError,
            message=(
                f"Key {key} begins with non-ascii letter; thus making it"
                " invalid for a configuration key."
            ),
        )

    # We ensure that the case of the key is upper case, and more importantly,
    # unique in case.
    upper_key = underscore_key.casefold().upper()

    # The current stage of the key is sanitized.
    sanitized_key = upper_key
    return sanitized_key


def sanitize_configuration_value(value: object) -> int | float | str:
    """Sanitize only the configuration value to a string.

    Value sanitization ensures just three properties:

    - The value in question can be serialized to and from a numeric or string.
    - The value is not a dictionary.
    - The value string can fit on one line.

    We need to require strings because that is the format yaml ensures.

    Parameters
    ----------
    value : str
        The configuration value to sanitize.

    Returns
    -------
    sanitized_value : int, float, or str
        The value, sanitized.

    """
    # We need to make sure it is not a dictionary, else, that is likely nested
    # configurations.
    if isinstance(value, dict):
        logging.critical(
            critical_type=logging.ConfigurationError,
            message=(
                "Input value is a dictionary, this would lead to non-flat"
                " configurations and files."
            ),
        )

    # We need to make sure it can be turned into a string.
    try:
        value_str = str(value)
    except ValueError:
        logging.critical(
            critical_type=logging.ConfigurationError,
            message=f"Input value {value} cannot be turned to a string.",
        )

    # We have no real metric for it all fitting onto a line. But, we do just
    # give a warning if it is long.
    too_long_value = 80
    if len(value_str) > too_long_value:
        logging.warning(
            warning_type=logging.ConfigurationWarning,
            message=f"Configuration value {value_str} is very long.",
        )

    # Lastly, we figure out what is the best representation.
    if isinstance(value, int | float | str):
        sanitized_value = value
    else:
        # Maybe it is still a number?
        try:
            sanitized_value = float(value_str)
        except ValueError:
            # Nope, it is better to just use the string value.
            sanitized_value = value_str
    return sanitized_value


def apply_configuration(configuration: dict) -> None:
    """Apply the configuration, input structured as a dictionary.

    Note configuration files should be flat, there should be no nested
    configuration parameters.

    Applied configuration values are stored as attached variables to this
    module. Applying the configurations to this module's global namespace is
    the preferred method of applying the configuration. As these configurations
    will not change, they are constant like and thus can be accessed in a
    more Pythonic manner.

    Parameters
    ----------
    configuration : dict
        The configuration dictionary we are going to apply.

    Returns
    -------
    None

    """
    # Constants typically are all capitalized in their variable naming.
    capital_configuration = {
        keydex.upper(): valuedex for keydex, valuedex in configuration.items()
    }
    # Check that the configuration names were capitalized.
    for keydex, capital_keydex in zip(
        configuration.keys(),
        capital_configuration.keys(),
        strict=True,
    ):
        if keydex.casefold() != capital_keydex.casefold():
            logging.error(
                error_type=logging.ConfigurationError,
                message=(
                    "The following configuration keys differ on the case"
                    f" transformation: {keydex} -> {capital_keydex}"
                ),
            )
        if keydex != capital_keydex:
            logging.error(
                error_type=logging.ConfigurationError,
                message=(
                    "The keys of configuration parameters should be in all"
                    " capital letters. The following key is inappropriate:"
                    f" {keydex}"
                ),
            )
    # Applying it to the global space of this module only.
    globals().update(capital_configuration)


def read_configuration_file(filename: str) -> dict:
    """Read the configuration file and output a dictionary of parameters.

    Note configuration files should be flat, there should be no nested
    configuration parameters.

    Parameters
    ----------
    filename : str
        The filename of the configuration file, with the extension. Will raise
        if the filename is not the correct extension, just as a quick check.

    Returns
    -------
    configuration : dict
        The dictionary which contains all of the configuration parameters
        within it.

    """
    # Checking the extension is valid, just as a quick sanity check that the
    # configuration file is proper.
    config_extension = ("yaml", "yml")
    filename_ext = lezargus.library.path.get_file_extension(pathname=filename)
    if filename_ext not in config_extension:
        logging.error(
            error_type=logging.FileError,
            message=(
                "Configuration file does not have the proper extension, it"
                " should be a yaml file."
            ),
        )
    # Loading the configuration file.
    try:
        with open(filename, encoding="utf-8") as config_file:
            configuration = dict(
                yaml.load(config_file, Loader=yaml.SafeLoader),
            )
    except FileNotFoundError:
        # The file is not found, it cannot be opened.
        logging.critical(
            critical_type=logging.FileError,
            message=(
                "The following configuration filename does not exist:"
                f" {filename}"
            ),
        )

    # Double check that the configuration is flat as per the documentation
    # and expectation.
    for valuedex in configuration.values():
        if isinstance(valuedex, dict):
            # A dictionary implies a nested configuration which is not allowed.
            logging.error(
                error_type=logging.ConfigurationError,
                message=(
                    "The configuration file should not have any embedded"
                    " configurations, it should be a flat file. Please use the"
                    " configuration file templates."
                ),
            )

    # The configuration dictionary should be good.
    return configuration


def write_configuration_file(
    filename: str,
    configuration: dict,
    overwrite: bool = False,
) -> None:
    """Write a configuration file based on provided configurations.

    Note configuration files should be flat, there should be no nested
    configuration parameters. Moreover, we only write configurations present
    as default or as overwritten by the provided configuration. This function
    does not account for current live configurations.

    Parameters
    ----------
    filename : str
        The filename of the configuration file to be saved, with the extension.
        Will raise if the filename is not the correct extension.
    configuration : dict
        The configuration which we will save, along with any defaults present
        in the main configuration file.
    overwrite : bool
        If True, we overwrite the configuration file if already present.

    Returns
    -------
    None

    """
    # We need to sanitize the input configuration first.
    configuration = sanitize_configuration(configuration=configuration)

    # We also need the default configuration.
    default_configuration_path = lezargus.library.path.merge_pathname(
        directory=INTERNAL_MODULE_INSTALLATION_PATH,
        filename="configuration",
        extension="yaml",
    )
    default_configuration = read_configuration_file(
        filename=default_configuration_path,
    )
    # Applying any overwrites.
    writing_configuration = {**default_configuration, **configuration}

    # We want to preserve the comment information explaining the
    # configurations, so, instead of just dumping the YAML, we attempt to just
    # create a file inplace and manually change the required lines.
    with open(default_configuration_path, encoding="utf-8") as default_file:
        default_lines = [
            linedex.strip().rstrip("\n") for linedex in default_file.readlines()
        ]

    # Now we search through all lines, finding the needed fields we need to
    # replace.
    new_lines = []
    for linedex in default_lines:
        if len(linedex) == 0:
            # The line is blank, but, we still want to preserve the whitespace
            # in the file.
            new_lines.append("")
        elif linedex.startswith("#"):
            # The line is likely a comment string. We just take the whole
            # line without modification.
            new_lines.append(linedex)
        elif ":" in linedex:
            # The line is a `key : value` pair. We need to determine the key
            # and value. However, if the value itself contains ":", we do not
            # want to split it; the key cannot contain ":" due to the
            # convention.
            default_key, default_value = linedex.split(":", maxsplit=1)
            writing_key = default_key.strip()
            # Now we need to determine if the value is contained within the
            # writing configuration.
            writing_value = writing_configuration.get(
                writing_key,
                default_value,
            )
            writing_line = f"{writing_key} : {writing_value}"
            new_lines.append(writing_line)
        else:
            # Something went wrong, this line is abnormal.
            logging.error(
                logging.ConfigurationError,
                message=(
                    "Configuration line cannot be parsed for writing:"
                    f" {linedex}"
                ),
            )

    # We need to do a few checks for the configuration file path.
    config_extension = ("yaml", "yml")
    filename_ext = lezargus.library.path.get_file_extension(pathname=filename)
    if filename_ext not in config_extension:
        logging.error(
            error_type=logging.FileError,
            message=(
                "Configuration file does not have the proper extension, it"
                " should be a yaml file."
            ),
        )
    # We also check the directory and path.
    directory = lezargus.library.path.get_directory(pathname=filename)
    if not os.path.isdir(directory):
        # The directory of the file does not exist.
        logging.warning(
            warning_type=logging.FileWarning,
            message=(
                f"Saving filename directory {directory} does not exist,"
                " creating it."
            ),
        )
        os.makedirs(directory, exist_ok=True)
    # And we check if the file exists.
    if os.path.isfile(filename) and not overwrite:
        logging.critical(
            critical_type=logging.FileError,
            message=f"Configuration file {filename} already exists.",
        )

    # Finally, saving the file. We need to make our own new line characters.
    new_lines = [linedex + "\n" for linedex in new_lines]
    with open(filename, mode="w", encoding="utf-8") as new_file:
        new_file.writelines(new_lines)

    # All done.


def load_configuration_file(filename: str) -> None:
    """Load a configuration file, then apply it.

    Reads a configuration file, the applies it to the current configuration.
    Note configuration files should be flat, there should be no nested
    configuration parameters.

    Parameters
    ----------
    filename : str
        The filename of the configuration file, with the extension. Will raise
        if the filename is not the correct extension, just as a quick check.

    Returns
    -------
    None

    """
    # Loading a configuration is simply just reading the file, then applying
    # the configuration.
    configuration = read_configuration_file(filename=filename)
    apply_configuration(configuration=configuration)
    # Notifying that it was applied.
    logging.info(
        message=f"Configuration file {filename} was loaded and applied.",
    )


def create_configuration_file(
    filename: str,
    overwrite: bool = False,
) -> None:
    """Create a copy of the default configuration file to the given location.

    Parameters
    ----------
    filename : str
        The pathname or filename where the configuration file should be put
        to. If it does not have the proper yaml extension, it will be added.
    overwrite : bool, default = False
        If the file already exists, overwrite it. If False, it would raise
        an error instead.

    Returns
    -------
    None

    """
    # Check if the filename is already taken by something.
    if os.path.isfile(filename) and (not overwrite):
        logging.error(
            error_type=logging.FileError,
            message=(
                "Filename already exists, overwrite is False; file write is"
                f" skipped: {filename}"
            ),
        )

    # If the user did not provide a filename with the proper extension, add it.
    user_ext = lezargus.library.path.get_file_extension(pathname=filename)
    yaml_extensions = ("yaml", "yml")
    preferred_yaml_extension = yaml_extensions[0]
    if user_ext not in yaml_extensions:
        file_destination = lezargus.library.path.merge_pathname(
            filename=filename,
            extension=preferred_yaml_extension,
        )
    else:
        # Nothing needs to be done. The filename is fine.
        file_destination = filename

    # Copy the file over from the default location within this install.
    default_config_path = lezargus.library.path.merge_pathname(
        directory=lezargus.library.config.INTERNAL_MODULE_INSTALLATION_PATH,
        filename="configuration",
        extension="yaml",
    )
    shutil.copyfile(default_config_path, file_destination)


def update_configuration_file(
    filename: str,
    configuration: dict | None = None,
    new_filename: str | None = None,
) -> None:
    """Update a configuration file with new configuration parameters.

    This function updates a configuration with the provided configuration
    parameters. This is a file manipulation function and does not affect the
    actual runtime configuration. This function is mostly used to update
    outdated configuration files with newer templates while keeping any
    changed configurations.

    The configuration file is overwritten unless a new filename is provided.

    Parameters
    ----------
    filename : str
        The configuration file which we will update (either in place or with
        a copy). Configurations in the file are preserved unless overridden.
    configuration : dict, default = None
        Lat minute overriding configurations to apply to the updated
        configuration file. Keynames must match for an override to occur.
        By default, no overriding configurations. If None, we assume no
        overriding configurations.
    new_filename : str, default = None
        A new configuration filename to use if the original file is not to be
        overwritten.

    Returns
    -------
    None

    """
    # Defaults.
    configuration = {} if configuration is None else configuration
    # Reading the configuration file first.
    file_configuration = read_configuration_file(filename=filename)
    # And we sanitize it and the overwriting configurations
    file_configuration = sanitize_configuration(
        configuration=file_configuration,
    )
    configuration = sanitize_configuration(configuration=configuration)

    # We then merge the configurations.
    merge_configuration = {**file_configuration, **configuration}

    # And then we save the file. If a new filename has been provided, we use
    # that instead.
    if new_filename is not None:
        write_configuration_file(
            filename=new_filename,
            configuration=merge_configuration,
            overwrite=False,
        )
    else:
        # We need to allow overwriting of the original file, as that is the
        # design of this function.
        write_configuration_file(
            filename=filename,
            configuration=merge_configuration,
            overwrite=True,
        )


# Configuration/constant parameters which are otherwise not usually provided
# or must be provided at runtime with code.
###################

# The default path which this module is installed in. It is one higher than
# this file which is within the library module of the Lezargus install.
INTERNAL_MODULE_INSTALLATION_PATH = os.path.dirname(
    os.path.realpath(os.path.join(os.path.realpath(__file__), "..")),
)

# We need to get the actual directory of the data.
INTERNAL_MODULE_DATA_DIRECTORY = os.path.join(
    INTERNAL_MODULE_INSTALLATION_PATH,
    "data",
)
