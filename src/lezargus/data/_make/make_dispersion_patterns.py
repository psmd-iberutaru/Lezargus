"""Make functions to create the spectral dispersion classes.

This module is created to make interpolative spectral dispersion tables from
provided tables created from spot diagrams.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import lezargus
from lezargus.data._make import functionality


def make_spectre_dispersion_pattern(
    basename: str,
) -> hint.SpectreDispersionPattern:
    """Create a SPECTRE dispersion pattern from the file.

    Parameters
    ----------
    basename : str
        The base filename of the dispersion pattern file for SPECTRE.

    Returns
    -------
    spectre_dispersion : SpectreDispersionPattern
        The dispersion pattern class for the SPECTRE instrument, defined based
        on the inputted file data.

    """
    # Loading the dispersion file.
    filename = functionality.parse_basename(basename=basename)

    # And, just reading it in.
    dispersion_class = lezargus.library.container.SpectreDispersionPattern
    spectre_dispersion = dispersion_class.read_dispersion_table(
        filename=filename,
    )

    # All done.
    return spectre_dispersion
