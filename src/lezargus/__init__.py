"""Lezargus: The software package related to IRTF SPECTRE."""

# SPDX-FileCopyrightText: 2023-present Sparrow <psmd.iberutaru@gmail.com>
# SPDX-License-Identifier: MIT

import glob
import os
import sys
import uuid

# The library must be imported first as all other parts depend on it.
# Otherwise, a circular loop may occur in the imports. So, for autoformatting
# purposes, we need to tell isort/ruff that the library is a section all
# to itself.
from lezargus import library

# isort: split

# The data containers.
from lezargus import container

# Lastly, the main file. We only do this so that Sphinx correctly builds the
# documentation. (Though this too could be a misunderstanding.) Functionality
# of __main__ should be done via the command line interface.
from lezargus import __main__  # isort:skip

# Load the default configuration parameters. The user's configurations should
# overwrite these when supplied.
library.config.load_then_apply_configuration(
    filename=library.path.merge_pathname(
        directory=library.config.MODULE_INSTALLATION_PATH,
        filename="configuration",
        extension="yaml",
    ),
)

# Construct the default console and file-based logging functions. The file is
# saved in the package directory.
library.logging.add_stream_logging_handler(
    stream=sys.stderr,
    log_level=library.logging.LOGGING_INFO_LEVEL,
    use_color=library.config.LOGGING_STREAM_USE_COLOR,
)
# The default file logging is really a temporary thing (just in case) and
# should not kept from run to run. Moreover, if there are multiple instances
# of Lezargus being run, they all cannot use the same log file and so we
# encode a UUID tag.

# Adding a new file handler. We add the file handler first only so we can
# capture the log messages when we try and remove the old logs.
__DEFAULT_LEZARGUS_UNIQUE_HEX_IDENTIFIER = uuid.uuid1().hex
__DEFAULT_LEZARGUS_LOG_FILE_PATH = library.path.merge_pathname(
    directory=library.config.MODULE_INSTALLATION_PATH,
    filename="lezargus_" + __DEFAULT_LEZARGUS_UNIQUE_HEX_IDENTIFIER,
    extension="log",
)
library.logging.add_file_logging_handler(
    filename=__DEFAULT_LEZARGUS_LOG_FILE_PATH,
    log_level=library.logging.LOGGING_DEBUG_LEVEL,
)
# We try and remove all of the log files which currently exist, if we can.
# We make an exception for the one which we are going to use, we do not
# want to clog the log with it.
__old_log_files = glob.glob(
    library.path.merge_pathname(
        directory=library.config.MODULE_INSTALLATION_PATH,
        filename="lezargus*",
        extension="log",
    ),
    recursive=False,
)
for filedex in __old_log_files:
    if filedex == __DEFAULT_LEZARGUS_LOG_FILE_PATH:
        # We do not try to delete the current file.
        continue
    try:
        os.remove(filedex)
    except OSError:
        # The file is likely in use by another logger or Lezargus instance.
        # The deletion can wait.
        library.logging.info(
            message=(
                "The temporary log file {lfl} is currently in-use, we defer"
                " deletion.".format(lfl=filedex)
            ),
        )


# Load all of the data files for Lezargus.
library.data.initialize_data_files()
