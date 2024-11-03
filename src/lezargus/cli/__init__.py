"""Command line interactive paths and functions live here."""

from lezargus.cli import help_
from lezargus.cli import version

# isort: split

# The main execution function.
from lezargus.cli.__main__ import execute_primary_action
