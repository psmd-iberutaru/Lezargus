"""Lezargus: The software package related to IRTF SPECTRE."""

# SPDX-FileCopyrightText: 2023-present Sparrow <psmd.iberutaru@gmail.com>
# SPDX-License-Identifier: MIT

import os
import sys

# The library must be imported first as all other parts depend on it.
# Otherwise, a circular loop may occur in the imports.
from lezargus import library

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
# should not kept from run to run.
__DEFAULT_LEZARGUS_LOG_FILE_PATH = library.path.merge_pathname(
    directory=library.config.MODULE_INSTALLATION_PATH,
    filename="lezargus",
    extension="log",
)
if os.path.isfile(__DEFAULT_LEZARGUS_LOG_FILE_PATH):
    os.remove(__DEFAULT_LEZARGUS_LOG_FILE_PATH)
library.logging.add_file_logging_handler(
    filename=__DEFAULT_LEZARGUS_LOG_FILE_PATH,
    log_level=library.logging.LOGGING_DEBUG_LEVEL,
)
