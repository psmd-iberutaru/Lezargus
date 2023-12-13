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

# The initialization functionality.
# The data containers.
from lezargus import container
from lezargus import initialize

# Lastly, the main file. We only do this so that Sphinx correctly builds the
# documentation. (Though this too could be a misunderstanding.) Functionality
# of __main__ should be done via the command line interface.
from lezargus import __main__  # isort:skip
