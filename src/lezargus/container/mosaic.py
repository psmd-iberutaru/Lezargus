"""Mosaic data container.

This module and class primarily deals with a collection of data cubes pieced
together into a single combined mosaic. Unlike the previous containers, this
does not directly subclass Astropy NDData and instead acts as a collection of
other reduced spectral cubes and performs operations on it.
"""

import lezargus
from lezargus.library import hint
from lezargus.library import logging


class LezargusMosaic:
    """TODO."""

    def __init__(self: hint.Self) -> None:
        """Init doc."""
        __ = lezargus
        __ = hint
        __ = logging
        raise KeyboardInterrupt
