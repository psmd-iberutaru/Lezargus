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

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split

import contextlib
import os
import re
import uuid

import yaml

import lezargus
from lezargus.library import logging

# isort: split


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


def sanitize_configuration_value(value: hint.Any) -> int | float | str:
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


def assign_configuration(key: str, value: float | str) -> None:
    """Assign the configuration in lezargus.config.

    Parameters
    ----------
    key : str
        The configuration key value. If the key value does not exist in the
        main configuration, then an error is raised as it generally
        indicates a consistency error.
    value : int | float | str
        The value of the configuration to be set. Must be a simple type as
        configuration files should have pretty primitive types.

    Returns
    -------
    None

    """
    # We need to sanitize the key first.
    sanitize_key = sanitize_configuration_key(key=key)
    # And the value.
    sanitize_value = sanitize_configuration_value(value=value)

    # We then check the main configuration module for consistency before
    # applying it.
    if not hasattr(lezargus.config, sanitize_key):
        logging.critical(
            critical_type=logging.ConfigurationError,
            message=(
                "Lezargus configuration does not support the"
                f" {sanitize_key} key."
            ),
        )

    # Otherwise, we apply it.
    setattr(lezargus.config, sanitize_key, sanitize_value)


def apply_configuration(configuration: dict) -> None:
    """Apply the configuration, input structured as a dictionary.

    Note configuration files should be flat, there should be no nested
    configuration parameters.

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

    # We just sanitize the configuration.
    sanitize_config = sanitize_configuration(
        configuration=capital_configuration,
    )

    # Apply it to the configuration.
    for keydex, valuedex in sanitize_config.items():
        assign_configuration(key=keydex, value=valuedex)


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
            raw_configuration = dict(
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
    for valuedex in raw_configuration.values():
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

    # A final clean up of the configuration.
    configuration = sanitize_configuration(configuration=raw_configuration)

    # The configuration dictionary should be good.
    return configuration


def _convert_default_configuration_yaml(section: str = "ALL") -> list[str]:
    """Create a temporary configuration YAML, returning the file lines.

    The configuration file by default is a Python file to satisfy the type
    checker. However, most people will be using YAML files as it is safer.
    We convert the default configuration file to a YAML file, temporarily
    within the context manager, to manipulate, and provide to the user.

    The configuration file is split into sections based on the section tags.
    If a section label is provided, we only provide the configuration within
    the tags.

    Parameters
    ----------
    section : str, default
        The section label. We limit the YAML file to the section label subset.
        By default, we use ALL, the full file.

    Returns
    -------
    yaml_lines : str
        The file lines of the configuration YAML file.

    """
    # We get the default configuration Python file.
    config_py_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.config.INTERNAL_MODULE_INSTALLATION_PATH,
        filename="config",
        extension="py",
    )

    # We read in the Python file, care is needed to not load anything.
    with open(config_py_filename, encoding="utf-8") as config_py:
        all_config_py_lines = config_py.readlines()
        # We do not need the new line characters nor padding.
        all_config_py_lines = [
            linedex.removesuffix("\n").strip()
            for linedex in all_config_py_lines
        ]

    # By convention, we have tagged the beginning and end of the configuration
    # part of the Python file. So we only need those parts.
    clean_section = section.upper().strip()
    start_tag = f"# <BEGIN {clean_section}>"
    end_tag = f"# </END {clean_section}>"
    # We find the index parts.
    start_index = None
    end_index = None
    for index, linedex in enumerate(all_config_py_lines):
        if start_tag in linedex:
            start_index = index
        if end_tag in linedex:
            end_index = index
    # If we don't have either a start or end index, the tag provided is likely
    # incorrect.
    if start_index is None or end_index is None:
        start_index = 0
        end_index = -1
        logging.error(
            error_type=logging.InputError,
            message=(
                f"Configuration section tag {section} does not match any"
                " sections."
            ),
        )
    # We want the end tag as well.
    config_py_lines = all_config_py_lines[start_index : end_index + 1]

    # We will be testing the configuration lines to make sure they are
    # properly formatted before we attempt the conversion. The test using
    # regular expressions, it is better to compile it early.
    regex_test_pattern = r"[A-Z_]+ = [^=\n]*"
    regex_test = re.compile(regex_test_pattern)

    # The only real difference between the Python file and the YAML file is
    # the = and : as seperators. We leverage this fact and just change the
    # character.
    yaml_lines = []
    for linedex in config_py_lines:
        # Clearing out some cases where we should not change anything.
        if linedex.startswith("#"):
            # Line is a comment, no action taken. We just add it and move on.
            yaml_lines.append(linedex)
            continue
        if len(linedex) == 0:
            # Line is blank, no action can be taken. We just add it and move
            # on.
            yaml_lines.append(linedex)
            continue

        # We need to test that the configuration line is correct.
        if regex_test.match(linedex) is None:
            logging.critical(
                critical_type=logging.ConfigurationError,
                message=(
                    f"Configuration line {linedex} does not match the expected"
                    " format of a configuration setting."
                ),
            )
        else:
            # We just replace the = with a :.
            new_linedex = linedex.replace(R" = ", R" : ", 1)
            yaml_lines.append(new_linedex)
    # All done
    return yaml_lines


@contextlib.contextmanager
def _convert_default_configuration_file(
    section: str = "ALL",
) -> hint.Iterator[str]:
    """Create a temporary configuration YAML file, returning the path.

    See py:func:`_convert_default_configuration_yaml` for more information.

    Parameters
    ----------
    section : str, default
        The section label. We limit the YAML file to the section label subset.
        By default, we use ALL, the full file.

    Yeilds
    ------
    filename : str
        The filename of the default YAML file, we create it internally then
        delete it afterwards.

    """
    yaml_lines = _convert_default_configuration_yaml(section=section)

    # Now that we have the YAML format file, we can write it. We will not use
    # the temporary directory because it might not be configured yet.
    # We want to make sure there is no file conflict.
    yaml_random_filename = lezargus.library.path.merge_pathname(
        directory=lezargus.config.INTERNAL_MODULE_INSTALLATION_PATH,
        filename=f"temp_config_{uuid.uuid4()}",
        extension="yaml",
    )
    # Finally, saving the file. We need to make our own new line characters.
    raw_yaml_lines = [linedex + "\n" for linedex in yaml_lines]
    with open(yaml_random_filename, "xt", encoding="utf-8") as yaml_file:
        yaml_file.writelines(raw_yaml_lines)

    # We yeild to the user to execute the file as they may want to.
    yield yaml_random_filename

    # Finally, we clean up the file, it is not needed anymore and we don't
    # want it sticking around.
    os.remove(yaml_random_filename)


def write_configuration_file(
    filename: str,
    configuration: dict[str, hint.Any] | None = None,
    section: str = "ALL",
    overwrite: bool = False,
) -> None:
    """Write a configuration file based on provided configuration.

    Note configuration files should be flat, there should be no nested
    configuration parameters. Moreover, we only write configurations present
    as default or as overwritten by the provided configuration, within the
    section tag as provided. This function does not account for current live
    configurations.

    Parameters
    ----------
    filename : str
        The filename of the configuration file to be saved, with the extension.
        An error will be provided if the extension is not a correct extension.
    configuration : dict, default = None
        The configuration which we will save, along with any defaults present
        in the main configuration file. If None, only defaults are saved.
    section : str
        The section label for us to reduce the scope of the configuration
        file we will be writing.
    overwrite : bool
        If True, we overwrite the configuration file if already present.

    Returns
    -------
    None

    """
    # We need to sanitize the input configuration first.
    configuration = {} if configuration is None else configuration
    configuration = sanitize_configuration(configuration=configuration)

    # We also need the default configuration.
    default_configuration = get_default_configuration(section=section)

    # Applying any overwrites.
    writing_configuration = {**default_configuration, **configuration}

    # We want to preserve the comment information explaining the
    # configurations, so, instead of just dumping the YAML, we attempt to just
    # create a file inplace and manually change the required lines.
    default_lines = _convert_default_configuration_yaml(section=section)

    # Now we search through all lines, finding the needed fields we need to
    # replace.
    writing_lines = []
    for linedex in default_lines:
        # If the line is blank, we want to keep it blank in the write out.
        if len(linedex) == 0:
            writing_lines.append("")
            continue
        # If the line is a comment, it is likely documentation so we keep it
        # as is.
        if linedex.startswith("#"):
            writing_lines.append(linedex)
            continue

        # If it is a `key : value` pair, we need to determine the key and value
        # and replace it with the new one if needed.
        if R" : " in linedex:
            # We do not want to split more than the key value pair itself. The
            # key should never have : in it.
            default_key, default_value = linedex.split(R" : ", maxsplit=1)
            writing_key = default_key.strip()
            # Now we need to determine if the value is contained within the
            # writing configuration.
            writing_value = writing_configuration.get(
                writing_key,
                default_value,
            )
            temp_writing_line = f"{writing_key} : {writing_value}"
            writing_lines.append(temp_writing_line)

        # If a configuration line does not meet any of the above, it is not
        # parsable for writing.
        logging.error(
            logging.ConfigurationError,
            message=(
                f"Configuration line cannot be parsed for writing: {linedex}"
            ),
        )

    # We need to do a few checks for the configuration file path.
    config_extension = ("yaml", "yml")
    filename_ext = lezargus.library.path.get_file_extension(pathname=filename)
    if filename_ext not in config_extension:
        logging.error(
            error_type=logging.FileError,
            message=(
                f"Configuration filename has extension {filename_ext}, not a"
                " YAML extension (yml or yaml)."
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
    yaml_lines = [linedex + "\n" for linedex in writing_lines]
    with open(filename, mode="w", encoding="utf-8") as new_file:
        new_file.writelines(yaml_lines)


def get_default_configuration(section: str = "ALL") -> dict:
    """Get the default configuration dictionary.

    Parameters
    ----------
    section : str
        The section label for us to reduce the scope of the default
        configuration provided

    Returns
    -------
    default_configuration : dict
        The total default configuration dictionary.

    """
    # Loading the default configuration
    with _convert_default_configuration_file(section=section) as default:
        default_configuration = read_configuration_file(filename=default)
    return default_configuration


def create_configuration_file(
    filename: str,
    section: str = "ALL",
    overwrite: bool = False,
) -> None:
    """Create a copy of the default configuration file to the given location.

    Parameters
    ----------
    filename : str
        The filename of the new configuration file to be saved, with the
        extension. An error will be provided if the extension is not a
        correct extension.
    section : str
        The section label for us to reduce the scope of the configuration
        file we will be writing.
    overwrite : bool, default = False
        If the file already exists, overwrite it. If False, it would raise
        an error instead.

    Returns
    -------
    None

    """
    # This really is just the same as writing a configuration file, just with
    # no configuration changes from the default.
    write_configuration_file(
        filename=filename,
        configuration=None,
        section=section,
        overwrite=overwrite,
    )


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


def save_configuration_file(
    filename: str,
    section: str = "ALL",
    overwrite: bool = False,
) -> None:
    """Save the current live configuration to a configuration file.

    This function saves the current configuration to a configuration file.
    The entire configuration file is provided by default to replicate
    the current settings, but a section tag may be provided.

    Parameters
    ----------
    filename : str
        The filename of the configuration file to be saved, with the
        extension. An error will be provided if the extension is not a
        correct extension.
    section : str
        The section label for us to reduce the scope of the configuration
        file we will be saving.
    overwrite : bool, default = False
        If the file already exists, overwrite it. If False, it would raise
        an error instead.

    Returns
    -------
    None

    """
    # We need to export the current configuration. Extracting all of the
    # attributes ought to be fine but we need to sort though it after.
    raw_state = vars(lezargus.config)

    # We get the parts of the configuration that we actually need.
    default_configuration = get_default_configuration(section=section)

    # We cycle through the raw state, extracting the required configurations
    # from it.
    live_configuration = {
        keydex: raw_state.get(keydex, valuedex)
        for keydex, valuedex in default_configuration.items()
    }

    # Finally, we save the live configuration.
    write_configuration_file(
        filename=filename,
        configuration=live_configuration,
        section=section,
        overwrite=overwrite,
    )
