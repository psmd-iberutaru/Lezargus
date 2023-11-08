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


def initialize() -> None:
    """Initialize the Lezargus module and its parts.

    This initialization function should be the very first thing that is done
    when the module is loaded. However, we create this function (as opposed to
    doing it on load) to be explicit on the load times for the module, to
    avoid circular dependencies, and to prevent logging when only importing
    the module.

    The order of the initialization is important.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Load in the default configuration file.
    library.config.initialize_default_configuration()

    # Load the logging outputs.
    library.logging.initialize_default_logging_outputs()

    # All of the initializations below have logging.

    # Load all of the data files for Lezargus.
    library.data.initialize_data_files()
