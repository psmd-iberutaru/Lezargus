"""Lezargus: The software package related to IRTF SPECTRE."""

# SPDX-FileCopyrightText: 2022-present Sparrow <psmd.iberutaru@gmail.com>
# SPDX-License-Identifier: MIT


# A circular loop may occur in the imports. So, for autoformatting
# purposes, we need to tell isort/ruff that the library is a section all
# to itself.

# isort: split
# The library and configuration must be imported first as all other parts
# depend on it.
from lezargus import config
from lezargus import library

# isort: split
# Data files and objects, we need the library to create these objects.
from lezargus import data
from lezargus import simulator

# isort: split
# The initialization functionality.
from lezargus import initialize
from lezargus import terminate

# isort: split
# User-based functionality, the actual commands which call the above functions.
from lezargus import cli

# Lastly, the main file. We only do this so that Sphinx correctly builds the
# documentation. (Though this too could be a misunderstanding.) Functionality
# of __main__ should be done via the command line interface.
from lezargus import __main__  # isort:skip
