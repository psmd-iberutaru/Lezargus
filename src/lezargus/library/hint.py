"""A collection of types for linting.

These are redefinitions and wrapping variables for type hints. Its purpose
is for just uniform hinting types.

This should only be used for types which are otherwise not native and would
require an import, including the typing module. The whole point of this is to
be a central collection of types for the purpose of type hinting.

This module should never be used for anything other than hinting. Use proper
imports to access these classes. Otherwise, you will likely get circular
imports and other nasty things.
"""

# pylint: disable=W0611,W0614

from argparse import ArgumentParser
from argparse import Namespace
from collections.abc import Callable
from subprocess import CompletedProcess
from typing import Any
from typing import Literal
from typing import Self

# Astropy imports.
from astropy.io.fits import FITS_rec
from astropy.io.fits import Header
from astropy.io.fits.card import Undefined
from astropy.table import Row
from astropy.table import Table
from astropy.units import Quantity
from astropy.units import Unit
from astropy.wcs import WCS

# Matplotlib imports.
from matplotlib.backend_bases import MouseEvent

# Arrays. We use ndarray instead as ArrayLike casts a rather larger union
# in the documentation.
from numpy.typing import NDArray

# The GUI windows and other imports.
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

# And the windows within the widgets.
from PySide6.QtWidgets import QMainWindow
from PySide6.QtWidgets import QWidget

# Containers...
from lezargus.library.container import AtmosphereSpectrumGenerator
from lezargus.library.container import DetectorArray
from lezargus.library.container import LezargusContainerArithmetic
from lezargus.library.container import LezargusCube
from lezargus.library.container import LezargusImage
from lezargus.library.container import LezargusMosaic
from lezargus.library.container import LezargusSpectrum
from lezargus.library.container import PhotometricABFilter
from lezargus.library.container import PhotometricVegaFilter
from lezargus.library.container import SpectreDispersionPattern

# Lezargus aliases.
# Library things.
from lezargus.library.interpolate import Generic1DInterpolate
from lezargus.library.interpolate import Spline1DInterpolate

# Simulators...
from lezargus.simulator import AtmosphereSimulator
from lezargus.simulator import IrtfTelescopeSimulator
from lezargus.simulator import TargetSimulator
