"""Mosaic data container.

This module and class primarily deals with a collection of data cubes pieced
together into a single combined mosaic. Unlike the previous containers, this
does not directly subclass Astropy NDData and instead acts as a collection of
other reduced spectral cubes and performs operations on it.
"""

# isort: split
# Import required to remove circular dependencies from type checking.
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lezargus.library import hint
# isort: split


import lezargus
from lezargus.library import logging


class LezargusMosaic:
    """TODO."""

    def __init__(self: LezargusMosaic) -> None:
        """Init doc."""
        __ = lezargus
        __ = logging
        raise KeyboardInterrupt

    def _hello(self: hint.Self) -> None:
        return None
