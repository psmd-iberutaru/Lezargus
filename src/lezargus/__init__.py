"""Lezargus: The software package related to IRTF SPECTRE."""

# SPDX-FileCopyrightText: 2022-present Sparrow <psmd.iberutaru@gmail.com>
# SPDX-License-Identifier: MIT

# The library must be imported first as all other parts depend on it.
# Otherwise, a circular loop may occur in the imports. So, for autoformatting
# purposes, we need to tell isort/ruff that the library is a section all
# to itself.
from lezargus import library

# isort: split

# These are library-like functions that are separate from the library.
from lezargus import container

# The initialization functionality.
from lezargus import initialize

# isort:split
# User-based functionality, the actual classes which call the above functions.
from lezargus import simulator

# Lastly, the main file. We only do this so that Sphinx correctly builds the
# documentation. (Though this too could be a misunderstanding.) Functionality
# of __main__ should be done via the command line interface.
from lezargus import __main__  # isort:skip
