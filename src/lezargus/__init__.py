"""Lezargus: The software package related to IRTF SPECTRE."""

# SPDX-FileCopyrightText: 2023-present Sparrow <psmd.iberutaru@gmail.com>
# SPDX-License-Identifier: MIT


# The library must be imported first as all other parts depend on it.
# Otherwise, a circular loop may occur in the imports.
from lezargus import library

# Lastly, the main file. We only do this so that Sphinx correctly builds the
# documentation. (Though this too could be a misunderstanding.) Functionality
# of __main__ should be done via the command line interface.
from lezargus import __main__ #isort:skip

# Load the default configuration parameters. The user's configurations should
# overwrite these when supplied.
library.config.load_then_apply_configuration(
    filename=library.path.merge_pathname(
        directory=library.config.MODULE_INSTALLATION_PATH,
        filename="configuration",
        extension="yaml",
    ),
)
